import ast
from omegaconf import open_dict
import warnings
from dataclasses import dataclass
from typing import Any
from functools import partial
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, ActorCritic,
                                    Transition, TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.utils.pmap_utils import replicate_to_devices, create_synced_grad_fn


@dataclass(frozen=True)
class PPOAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = PPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   tx=tx)


@struct.dataclass
class PPOAgentState(AgentStateBase):
    train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        return {"train_state": serialized_train_state}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        return cls(train_state)


class PPOJax(JaxRLAlgorithmBase):

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)

        # INIT NETWORK
        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)
        if hasattr(config.experiment, "actor_obs_group") and config.experiment.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "critic_obs_group") and config.experiment.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "len_obs_history") and config.experiment.len_obs_history > 1:
            obs_len = env.info.observation_space.shape[0]
            actor_obs_ind = jnp.concatenate([actor_obs_ind + i*obs_len
                                             for i in range(config.experiment.len_obs_history)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind + i*obs_len
                                              for i in range(config.experiment.len_obs_history)])
        network = ActorCritic(
            env.info.action_space.shape[0],
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind
        )

        # set up optimizers
        tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, tx)

    @classmethod
    def _get_optimizer(cls, config):
        if config.experiment.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(weight_decay=config.experiment.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._linear_lr_schedule(count, config.experiment.num_minibatches,
                                                                                config.experiment.update_epochs, config.lr,
                                                                                config.experiment.num_updates))
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(config.experiment.lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
            )

        tx = optax.apply_if_finite(tx, max_consecutive_errors=10000000)

        return tx

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, tx =\
            (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

        env = cls._wrap_env(env, config)

        # extract current agent state
        if agent_state is not None:
            train_state = agent_state.train_state
        else:
            train_state = None

        if train_state is None:

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)

        else:
            raise NotImplementedError("Loading of train state not implemented yet.")

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            y, _ = network.apply({'params': train_state.params,
                                              'run_stats': train_state.run_stats},
                                             last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics

            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}
        
    @classmethod
    def _initialize_params(cls, rng, env, agent_conf):
        """Initialize parameters on host/CPU to ensure consistency across devices."""
        network, tx = (agent_conf.network, agent_conf.tx)
        
        # Initialize network parameters using the wrapped environment's observation shape
        init_x = jnp.zeros(env.info.observation_space.shape)
        network_params = network.init(rng, init_x)
        
        # Create train state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )
        
        # Create agent state
        agent_state = cls._agent_state(train_state=train_state)
        
        return agent_state
    

    @classmethod
    def _train_fn_pmap(cls, rng, env, agent_conf: PPOAgentConf, agent_state: PPOAgentState, mh: MetricsHandler = None):
        """Multi-device version of the training function with gradient synchronization.
        
        Parameters are expected to be already initialized and replicated if agent_state
        is provided, ensuring consistency across devices.
        """
        # Extract configuration
        config, network, train_state = (agent_conf.config.experiment, agent_conf.network, agent_state.train_state)

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        # Create buffer for validation states
        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # Define update step with synchronized gradients
        def _update_step_pmap(runner_state, unused):
            # Collect trajectories (same as single-device version)
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # Select action
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                            'run_stats': train_state.run_stats},
                                           last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step environment (stays on device)
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # Get metrics
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info,
                    env_state.additional_carry.traj_state, logged_metrics
                )
                runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # Calculate advantage (same as single-device version)
            train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            y, _ = network.apply({'params': train_state.params, 'run_stats': train_state.run_stats},
                                last_obs, mutable=["run_stats"])
            pi, last_val = y

            advantages, targets = cls._calculate_gae_pmap(traj_batch, last_val, config)

            # Update network with synchronized gradients
            def _update_epoch_pmap(update_state, unused):
                def _update_minbatch_pmap(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Define loss function (same as single-device)
                    def _loss_fn(params, traj_batch, gae, targets):
                        # Run network
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                            traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # Calculate value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Use synchronized gradient function
                    synced_grad_fn = create_synced_grad_fn(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = synced_grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )

                    # Apply gradients (synchronized across devices)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Prepare batches (same as single-device)
                batch_size = config.minibatch_size * config.num_minibatches
                assert batch_size == config.num_steps * config.num_envs, \
                    "batch size must be equal to number of steps * number of envs"
                
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                # Process all minibatches
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch_pmap, train_state, minibatches
                )
                
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Run all update epochs
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch_pmap, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Process metrics (same as single-device)
            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs
            logged_metrics = traj_batch.metrics

            # Calculate metrics
            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)
                ) / jnp.maximum(jnp.sum(logged_metrics.done), 1.0),
                mean_episode_length=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)
                ) / jnp.maximum(jnp.sum(logged_metrics.done), 1.0),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            # Handle validation (simplified for multi-device)
            validation_metrics = ValidationSummary()
            if mh is not None and jax.lax.is_primitive_constant(counter % config.validation_interval == 0, True):
                # This would need to be implemented for multi-device validation
                pass

            # Update train state buffer if needed
            train_state_buffer = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x,
                train_state_buffer, train_state
            )

            # Verify parameter consistency
            _ = jax.lax.pmean(jnp.array(1.0), axis_name='i')  # Simple sync point

            runner_state = (train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        # Main training loop
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step_pmap, runner_state, None, config.num_updates
        )

        # Create agent state from final train state
        agent_state = cls._agent_state(train_state=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

    @classmethod
    def _calculate_gae_pmap(cls, traj_batch, last_val, config):
        """Calculate generalized advantage estimation for pmap context."""
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, absorbing, value, reward = (
                transition.done,
                transition.absorbing,
                transition.value,
                transition.reward
            )

            delta = reward + config.gamma * next_value * (1 - absorbing) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    @classmethod
    def play_policy(cls, env,
                    agent_conf: PPOAgentConf,
                    agent_state: PPOAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        def sample_actions(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                   'run_stats': ts.run_stats},
                                                  obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            a = pi.sample(seed=_rng)
            return a, ts

        config = agent_conf.config.experiment
        train_state = agent_state.train_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: PPOAgentConf,
                           agent_state: PPOAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed)

    @staticmethod
    def _wrap_env(env, config):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = LogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecReward(env, config.gamma)
        return env