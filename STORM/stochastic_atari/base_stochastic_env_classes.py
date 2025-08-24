import numpy as np
from typing import Dict, Any, Type
from gymnasium.core import ActionWrapper, ObservationWrapper, Wrapper
from stochastic_atari.utils import crop_obs_mode
import logging

logger = logging.getLogger(__name__)

class InternalGymWrapper(Wrapper):
    """Wrapper to set original ram state"""

    def __init__(self, env):
        super().__init__(env)
        self.env.unwrapped.original_ram_state = self.env.unwrapped.ale.getRAM()


class ActionDependentStochasticityWrapper(ActionWrapper):
    """Wrapper that implements action dependent stochasticity.
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.prob = config['stochastic_action_prob']

    def action(self, action):
        logger.debug(f"using action() method of ActionDependentStochasticityWrapper")
        if np.random.random() < self.prob:
            # Choose random action
            return self.env.action_space.sample()
        else:
            # Use predicted action
            return action


class ActionIndependentRandomStochasticityWrapper(ActionWrapper):
    """Wrapper that implements action independent random stochasticity.
    
    Note: This is an abstract class that needs to be implemented for each environment.

    Modes:
       mode '0': No stochasticity.
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.mode = config['mode']
        self.prob = config['random_stochasticity_prob']

    def step(self, action):
        if self.mode == '0':
            return self.env.step(action)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")


class ActionIndependentConceptDriftWrapper(ActionWrapper):
    """
    Wrapper that implements action independent concept drift in the environment.

    Modes:
        temporal_mode 'sudden': The environment concept (stochasticity type) switches suddenly after a fixed number of steps (temporal_threshold).
        temporal_mode 'cyclic': The environment concept alternates cyclically every temporal_threshold steps between the original and secondary concept.

    Args:
        env: The environment to wrap.
        config: Dictionary with keys:
            - 'temporal_mode': 'sudden' or 'cyclic'
            - 'temporal_threshold': Number of steps before switching concepts
            - 'secondary_concept_type': The stochasticity type to switch to
        StochasticEnv_instance: An instance that manages the stochastic environment and can update its type.
    """
    def __init__(self, env, config, StochasticEnv_instance):
        super().__init__(env)
        self._step_count = 0
        self._noskip_step_count = 0 # for skip correction in step count where skip is not part of original env
        self.current_cycle = 0
        self.temporal_mode = config['temporal_mode']
        self.temporal_threshold = config['temporal_threshold']
        self.secondary_concept_type = config['secondary_concept_type']
        self.skip = config.get('skip', 1) # since skip is not part of original env, we need to add it here
        self.StochasticEnv_instance = StochasticEnv_instance

    def update_env_concept(self):
        print(f"updating env concept to stochasticity type: {self.secondary_concept_type}")
        if self.secondary_concept_type == 3:
            raise RecursionError("`Concept drift` is not supported for secondary concept")
        if self.secondary_concept_type == 4:
            raise ValueError("`Partial observation` is the initial concept")
        self.StochasticEnv_instance.type = self.secondary_concept_type
        self.env = self.StochasticEnv_instance.get_env(self.env)

    def revert_env_concept(self):
        print(f"reverting to original concept")
        # if hasattr(self.env, 'manipulation'):
        #     print(f"reverting to original obs")
        #     self.env.manipulation = False
        self.env = self.env.env

    def get_cycle(self):
        if self.temporal_mode == 'cyclic':
            return 0 if (self._step_count // self.temporal_threshold) % 2 == 0 else 1
        else:
            return 0

    def step(self, action):

        if self.temporal_mode == 'sudden':
            if self._step_count == self.temporal_threshold - 1:
                self.update_env_concept()

        elif self.temporal_mode == 'cyclic': # currently bi-cyclic
            _cycle = self.get_cycle()
            if _cycle != self.current_cycle:
                if self.current_cycle == 0:
                    self.update_env_concept()
                else:
                    self.revert_env_concept()
                self.current_cycle = _cycle
        else:
            raise NotImplementedError(f"Mode {self.temporal_mode} not implemented")

        obs, reward, done, truncated, info = self.env.step(action)
        self._noskip_step_count += 1
        if self._noskip_step_count == self.skip:
            self._step_count += 1
            self._noskip_step_count = 0
        return obs, reward, done, truncated, info

    def reset(self, *args, **kwargs):
        logger.debug(f"using reset() method of ActionIndependentConceptDriftWrapper")
        obs = self.env.reset(*args, **kwargs)
        self._step_count = 0
        self._noskip_step_count = 0
        return obs


class PartialObservationWrapper(ObservationWrapper):
    """Custom wrapper that modifies observations"""

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.manipulation = True # if false, returns original _screen behavior

    def _manipulate_obs(self, array):
        """Modify the observation"""

        # add a random probability to apply the modification
        if np.random.rand() < self.config['prob']:
            if self.config['type'] == 'blackout':
                self._blackout_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'crop':
                self._crop_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'ram':
                array[:] = self._ram_obs_mode(self.env, self.config['mode'])
            else:
                raise NotImplementedError(f"Type {self.config['type']} not implemented")

    def _crop_obs_mode(self, array, mode) -> None:
        crop_obs_mode(array, mode)

    def _blackout_obs_mode(self, array, mode) -> None:
        raise NotImplementedError("Blackout observation modification not implemented")

    def _ram_obs_mode(self, env, mode) -> np.ndarray:
        raise NotImplementedError("RAM observation modification not implemented")

    def observation(self, observation):
        if self.manipulation:
            self._manipulate_obs(observation)
        return observation



class StochasticEnv:
    """
    StochasticEnv class that uses game-specific registries.

    Environment types:
    0: Deterministic Env - No stochasticity or partial observability applied.
    1: Intrinsic Stochastic Env (action-dependent) - Stochasticity based on agent's actions.
    2: Intrinsic Stochastic Env (action-independent-random) - Random stochasticity effects.
    3: Intrinsic Stochastic Env (action-independent-concept-drift) - Concept drift over time.
    4: Partially observed Env (state-variable-different-repr) - Different state representation.
    5: Partially observed Env (state-variable-missing) - Missing state variables.
    """

    SAMPLE_CONFIG = {
        'intrinsic_stochasticity': {
            'action_dependent': {
                'stochastic_action_prob': 1.0,
                },
            'action_independent_random': {
                'mode': '2',
                'random_stochasticity_prob': 0.25,
            },
            'action_independent_concept_drift': {
                'temporal_mode': 'cyclic',
                'temporal_threshold': 5,
                'secondary_concept_type': 1,
            },
        },
        'partial_observation': {
                'type': 'crop',
                'mode': '2',
                'prob': 0.75,
        },
    }

    def __init__(self, type: int, config: Dict[str, Any], wrapper_registry: Dict[str, Type] = None):
        self.type = type
        self.config = config
        if wrapper_registry is None:
            self.wrapper_registry = {
                    'action_dependent': ActionDependentStochasticityWrapper,
                    'action_independent_random': ActionIndependentRandomStochasticityWrapper,
                    'action_independent_concept_drift': ActionIndependentConceptDriftWrapper,
                    'partial_observation': PartialObservationWrapper,
                }
        else:
            self.wrapper_registry = wrapper_registry

    def get_env(self, env):
        """Apply the appropriate wrapper based on type"""
        env = InternalGymWrapper(env)
        if self.type == 0:
            raise NotImplementedError

        elif self.type == 1:
            wrapper_class = self.wrapper_registry.get('action_dependent')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action dependent wrapper not registered")
            return wrapper_class(env, config=self.config['intrinsic_stochasticity']['action_dependent'])

        elif self.type == 2:
            wrapper_class = self.wrapper_registry.get('action_independent_random')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action independent random wrapper not registered")
            return wrapper_class(env, config=self.config['intrinsic_stochasticity']['action_independent_random'])

        elif self.type == 3:
            wrapper_class = self.wrapper_registry.get('action_independent_concept_drift')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action independent concept drift wrapper not registered")
            return wrapper_class(env, config=self.config['intrinsic_stochasticity']['action_independent_concept_drift'], StochasticEnv_instance=self)

        elif self.type == 4:
            return env  # Default, no wrapper

        elif self.type == 5:
            wrapper_class = self.wrapper_registry.get('partial_observation')
            if wrapper_class is None:
                raise ModuleNotFoundError("Partial observation wrapper not registered")
            return wrapper_class(env, config=self.config['partial_observation'])

        else:
            raise ValueError(f"Unknown stochasticity type: {self.type}")
