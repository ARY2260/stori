import numpy as np
from typing import Dict, Any, Type
from gymnasium.core import ActionWrapper, Wrapper
from stochastic_atari.utils import crop_obs_mode
import logging

logger = logging.getLogger(__name__)


class CutomGymnasiumWrapper(Wrapper):
    """Wrapper to convert gymnasium API back to old gym API"""
    def __init__(self, env):
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None
    
    def reset(self):
        return self.env.reset()


class ActionDependentStochasticityWrapper(ActionWrapper):
    """Wrapper that implements action dependent stochasticity in internal ale _env.
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
    """Wrapper that implements action independent random stochasticity in internal ale _env.
    
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


class ActionIndependentConceptDriftWrapper(CutomGymnasiumWrapper):
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
        self.current_cycle = 0
        self.temporal_mode = config['temporal_mode']
        self.temporal_threshold = config['temporal_threshold']
        self.secondary_concept_type = config['secondary_concept_type']
        self.StochasticEnv_instance = StochasticEnv_instance

    def update_env_concept(self):
        print(f"updating env concept to stochasticity type: {self.secondary_concept_type}")
        if self.secondary_concept_type == '2.2':
            raise RecursionError("`Concept drift` is not supported for secondary concept")
        if self.secondary_concept_type == '3.1':
            raise ValueError("`Partial observation - 3.1` is the initial concept")
        self.StochasticEnv_instance.type = self.secondary_concept_type
        self.env = self.StochasticEnv_instance.get_env(self.env)

    def revert_env_concept(self):
        print(f"reverting to original concept")
        if hasattr(self.env, 'manipulation'):
            print(f"reverting to original screen")
            self.env.manipulation = False
        if hasattr(self.env, 'env'):
            self.env = self.env.env # for type 5
        else:
            self.env._env = self.env._env.env # for type 1, 2

    def get_cycle(self):
        if self.temporal_mode == 'cyclic':
            return 0 if (self._step_count // self.temporal_threshold) % 2 == 0 else 1
        else:
            return 0

    def step(self, action):

        if self.temporal_mode == 'sudden':
            if self._step_count == self.temporal_threshold - 1:
                self.update_env_concept()
                self.current_cycle = 1

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

        obs, reward, done, info = self.env.step(action)
        self._step_count += 1
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        logger.debug(f"using reset() method of ActionIndependentConceptDriftWrapper")
        if self.current_cycle != 0:
            print(f"resetting to original concept")
            self.revert_env_concept()
            self.current_cycle = 0

        obs = self.env.reset(*args, **kwargs)
        self._step_count = 0
        return obs


class PartialObservationWrapper(CutomGymnasiumWrapper):
    """Custom wrapper that modifies observations"""

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

        # Replace the _screen method from Atari env
        self.env._screen = self._screen
        self.manipulation = True # if false, returns original _screen behavior

    def _manipulate_screen(self, array):
        """Modify the screen in buffer"""

        # add a random probability to apply the modification
        if np.random.rand() < self.config['prob']:
            if self.config['type'] == 'blackout':
                self._blackout_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'crop':
                self._crop_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'ram':
                array_modified = self._ram_obs_mode(self.env._env, self.config['mode'])
                if array_modified is not None:
                    array[:] = array_modified
            else:
                raise NotImplementedError(f"Type {self.config['type']} not implemented")

    def _crop_obs_mode(self, array, mode) -> None:
        crop_obs_mode(array, mode)

    def _blackout_obs_mode(self, array, mode) -> None:
        raise NotImplementedError("Blackout observation modification not implemented")

    def _ram_obs_mode(self, env, mode) -> np.ndarray:
        raise NotImplementedError("RAM observation modification not implemented")

    def _screen(self, array):
        self.env._ale.getScreenRGB(array)
        if self.manipulation:
            self._manipulate_screen(array)


class StochasticEnv:
    """
    StochasticEnv class that uses game-specific registries.

    Environment types:
    0: Deterministic Env - No stochasticity or partial observability applied.
    1: Intrinsic Stochastic Env (action-dependent) - Stochasticity based on agent's actions.
    2.1: Intrinsic Stochastic Env (action-independent-random) - Random stochasticity effects.
    2.2: Intrinsic Stochastic Env (action-independent-concept-drift) - Concept drift over time.
    3.1: Partially observed Env (state-variable-different-repr) - Different state representation.
    3.2: Partially observed Env (state-variable-missing) - Missing state variables.
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
                'secondary_concept_type': '3.2',
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
        if self.type == 0:
            raise NotImplementedError

        elif self.type == '1':
            wrapper_class = self.wrapper_registry.get('action_dependent')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action dependent wrapper not registered")
            env._env = wrapper_class(env._env, config=self.config['intrinsic_stochasticity']['action_dependent'])
            return env

        elif self.type == '2.1':
            wrapper_class = self.wrapper_registry.get('action_independent_random')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action independent random wrapper not registered")
            env._env = wrapper_class(env._env, config=self.config['intrinsic_stochasticity']['action_independent_random'])
            return env

        elif self.type == '2.2':
            wrapper_class = self.wrapper_registry.get('action_independent_concept_drift')
            if wrapper_class is None:
                raise ModuleNotFoundError("Action independent concept drift wrapper not registered")
            return wrapper_class(env, config=self.config['intrinsic_stochasticity']['action_independent_concept_drift'], StochasticEnv_instance=self)

        elif self.type == '3.1':
            return env  # Default, no wrapper

        elif self.type == '3.2':
            wrapper_class = self.wrapper_registry.get('partial_observation')
            if wrapper_class is None:
                raise ModuleNotFoundError("Partial observation wrapper not registered")
            return wrapper_class(env, config=self.config['partial_observation'])

        else:
            raise ValueError(f"Unknown stochasticity type: {self.type}")
