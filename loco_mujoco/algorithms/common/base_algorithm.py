import pickle
from pathlib import Path
from dataclasses import dataclass

import jax
from flax import struct

from loco_mujoco.utils import MetricsHandler


@dataclass(frozen=True)
class AgentConfBase:
    """
    Abstract base class for *static* agent configuration.
    Any subclass must implement the serialize and from_dict methods.
    """

    def serialize(self):
        """Serialize the agent configuration."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """Create an instance of the configuration class from a dictionary."""
        raise NotImplementedError


@struct.dataclass
class AgentStateBase:
    """
    Abstract base class for *dynamic* agent state.
    Any subclass must implement the serialize and from_dict methods.
    """

    def serialize(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d, agent_conf):
        raise NotImplementedError


class JaxRLAlgorithmBase:

    """
    Base class defining the interface for all JAX RL algorithms.
    """

    # these class attributes must be set by subclasses
    _agent_conf: AgentConfBase
    _agent_state: AgentStateBase

    # prefix for all saved agent files
    _saved_agent_suffix = ".pkl"

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Initialize the agent configuration.
        """
        raise NotImplementedError

    @classmethod
    def build_train_fn(cls, env, agent_conf: AgentConfBase, mh: MetricsHandler = None):
        """ Returns the main train function of an RL algorithm used to train an agent from scratch. """
        return lambda rng_key: cls._train_fn(rng_key, env, agent_conf, mh=mh)
    
    @classmethod
    def build_pmap_train_fn(cls, env, agent_conf: AgentConfBase, num_devices=None, mh: MetricsHandler = None):
        """Returns a pmapped training function for multi-device training.
        
        Initializes parameters on one device first, then replicates to all devices to 
        ensure consistency from the beginning of training.
        """
        if num_devices is None:
            num_devices = jax.device_count()
        
        # Wrap the environment once
        config = agent_conf.config.experiment
        wrapped_env = cls._wrap_env(env, config)
        
        # Initialize parameters and agent state on CPU/host
        def train_wrapper(master_rng):
            # Initialize parameters on host/CPU
            init_rng, train_rng = jax.random.split(master_rng)
            
            # Pre-initialize agent_state
            agent_state = cls._initialize_params(init_rng, wrapped_env, agent_conf)
            
            # Generate a random key for each device
            device_rngs = jax.random.split(train_rng, num_devices)
            
            # Apply pmap to the training function
            p_train = jax.pmap(
                lambda rng: cls._train_fn_pmap(rng, wrapped_env, agent_conf, agent_state, mh),
                axis_name='i'
            )
            
            # Run training on all devices
            results = p_train(device_rngs)
            
            # Return results from first device (they should be identical)
            return jax.tree.map(lambda x: x[0], results)
        
        return train_wrapper

    @classmethod
    def _train_fn_pmap(cls, rng, env, agent_conf: AgentConfBase, agent_state: AgentStateBase = None, mh: MetricsHandler = None):
        """Multi-device version of the training function. Override in subclasses."""
        raise NotImplementedError("Multi-device training not implemented for this algorithm.")

    @classmethod
    def build_resume_train_fn(cls, env, agent_conf: AgentConfBase, mh: MetricsHandler = None):
        """ Returns the main train function of an RL algorithm used to resume training of an agent. """
        raise NotImplementedError("Resume training not implemented yet.")
        # todo: implement resume training, not working yet
        #return lambda rng_key, agent_state: (cls._train_fn(rng_key, env, agent_conf, agent_state, mh=mh))

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: AgentConfBase,
                  agent_state: AgentStateBase = None,
                  mh: MetricsHandler = None):
        """ Main training algorithm of an RL algorithm. """
        raise NotImplementedError

    @classmethod
    def play_policy(cls, train_state, env, config, n_envs, n_steps=None, record=False, key=None):
        raise NotImplementedError

    @classmethod
    def save_agent(cls, path, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """ Save the agent state to a file."""
        path = Path(path)
        path = path / (cls.__name__ + "_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        # serialize agent state
        serialized_state = cls.serialize(agent_conf, agent_state)
        # save agent state
        with open(path, 'wb') as file:
            pickle.dump(serialized_state, file)
        print(f"\nSaved agent to: {path}\n")
        return path

    @classmethod
    def load_agent(cls, path):
        """ Load the agent state from a file. """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != cls._saved_agent_suffix:
            raise ValueError(f'Not a {cls._saved_agent_suffix} file: {path}')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return cls.from_dict(data)

    @classmethod
    def serialize(cls, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """ Serialize conf and state of an agent. """
        serialized_agent_conf = agent_conf.serialize()
        serialized_agent_state = agent_state.serialize()
        return {"agent_conf": serialized_agent_conf,
                "agent_state": serialized_agent_state}

    @classmethod
    def from_dict(cls, d):
        """ Load conf and state of an agent from a dictionary. """
        agent_conf = cls._agent_conf.from_dict(d["agent_conf"])
        agent_state = cls._agent_state.from_dict(d["agent_state"], agent_conf)
        return agent_conf, agent_state

    @staticmethod
    def _wrap_env(env, config):
        """ Wrap the environment with the necessary wrappers. """
        raise NotImplementedError

    @classmethod
    def _linear_lr_schedule(cls, count, num_minibatches, update_epochs, lr, num_updates):
        frac = (
                1.0
                - (count // (num_minibatches * update_epochs))
                / num_updates
        )
        return lr * frac