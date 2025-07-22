import pytest
from jax import make_jaxpr

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, GAILJax, AMPJax
from loco_mujoco.utils import MetricsHandler

from test_conf import *

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

# Set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def test_PPO_Jax_build_train_fn(ppo_rl_config):

    config = ppo_rl_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # get initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    # build training function
    train_fn = PPOJax.build_train_fn(env, agent_conf)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))

        jaxpr = make_jaxpr(train_fn)(_rng)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


@pytest.mark.parametrize("algorithm", ("GAIL", "AMP"))
def test_Imitation_Jax_build_train_fn(algorithm, imitation_config):

    alg_cls = GAILJax if algorithm == "GAIL" else AMPJax

    config = imitation_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # create an expert dataset
    expert_dataset = env.create_dataset()

    # get initial agent configuration
    agent_conf = alg_cls.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)

    # setup metric handler (optional)
    mh = MetricsHandler(config, env) if config.experiment.validation.active else None

    # build training function
    train_fn = alg_cls.build_train_fn(env, agent_conf, mh=mh)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))

        jaxpr = make_jaxpr(train_fn)(_rng)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_PPO_pmap_compilation(ppo_rl_config):
    """Test that the pmapped training function compiles correctly."""
    
    config = ppo_rl_config
    
    # Use very small config for testing
    config.experiment.validation.num = 1
    config.experiment.num_updates = 2
    config.experiment.num_steps = 2
    config.experiment.num_envs = 32
    config.experiment.total_timesteps = config.experiment.num_envs * config.experiment.num_steps * config.experiment.num_updates
    
    # Get task factory and create environment
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # Get initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    # Build pmapped training function
    train_fn = PPOJax.build_pmap_train_fn(env, agent_conf)

    # Check function can be compiled
    try:
        rng = jax.random.PRNGKey(0)
        jax.make_jaxpr(train_fn)(rng)
    except Exception as e:
        pytest.fail(f"Failed to compile pmapped function: {e}")


def test_PPO_pmap_minimal_run(ppo_rl_config):
    """Run a minimal training to ensure the pmapped function works."""
    config = ppo_rl_config
    
    # Use very small config for testing
    config.experiment.validation.num = 1
    config.experiment.num_updates = 2
    config.experiment.num_steps = 2
    config.experiment.num_envs = 32
    config.experiment.total_timesteps = config.experiment.num_envs * config.experiment.num_steps * config.experiment.num_updates
    
    # Get task factory and create environment
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # Get initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    # Build pmapped training function
    num_devices = jax.device_count()
    train_fn = PPOJax.build_pmap_train_fn(env, agent_conf, num_devices=num_devices)

    # Run training
    rng = jax.random.PRNGKey(0)
    result = train_fn(rng)
    
    # Check for expected output structure
    assert "agent_state" in result
    assert "training_metrics" in result
    assert hasattr(result["training_metrics"], "mean_episode_return")