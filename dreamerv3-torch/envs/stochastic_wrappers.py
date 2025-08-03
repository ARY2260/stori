import numpy as np
from envs.post_processing_obs import blackout_obs_mode, crop_obs_mode
from envs.ram_modification_obs import ram_obs_modification_mode
from gymnasium.core import ActionWrapper
import gymnasium as gym

class CutomGymnasiumWrapper(gym.Wrapper):
    """Wrapper to convert gymnasium API back to old gym API"""
    def __init__(self, env):
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None
    
    def reset(self):
        return self.env.reset()

class PartialObservationWrapper(CutomGymnasiumWrapper):
    """Custom wrapper that modifies observations"""
    
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

        # Replace the _screen method from Atari env
        self.env._screen = self._screen
    
    def _manipulate_screen(self, array):
        """Modify the screen in buffer"""

        # add a random probability to apply the modification
        if np.random.rand() < self.config['prob']:
            if self.config['type'] == 'blackout':
                blackout_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'crop':
                crop_obs_mode(array, self.config['mode'])
            elif self.config['type'] == 'ram':
                array = ram_obs_modification_mode(self.env._env, self.config['mode'])
    
    def _screen(self, array):
        self.env._ale.getScreenRGB(array)
        self._manipulate_screen(array)


class ActionDependentStochasticityWrapper(ActionWrapper):
    """Wrapper that implements action dependent stochasticity in internal ale _env.
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.prob = config['stochastic_action_prob']

    def action(self, action):
        if np.random.random() < self.prob:
            # Choose random action
            return self.env.action_space.sample()
        else:
            # Use predicted action
            return action


class StochasticEnv:
    """
    Environment type docstrings:

    0: Deterministic Env
       - No stochasticity or partial observability applied.

    1: Intrinsic Stochastic Env (action-dependent)
       - Stochasticity is introduced based on the agent's actions.

    2: Intrinsic Stochastic Env (action-independent-random, Aleatoric)
       - Stochasticity is introduced independently of the agent's actions, e.g., random effects. # block not hit, regen

    3: Intrinsic Stochastic Env (action-independent-concept-drift)
       - Stochasticity is introduced by changing environment dynamics over time (concept drift). # within episode and combine sudden and cyclic

    4: Partially observed Env (state-variable-different-repr)
       - The environment state is partially observed by representing state variables differently.

    5: Partially observed Env (state-variable-missing)
       - The environment state is partially observed by omitting some state variables.
    """
    def __init__(self, type, config):
        self.type = type
        self.config = config

    def get_env(self, env):
        if self.type == 0:
            raise NotImplementedError
        elif self.type == 1:
            env._env = ActionDependentStochasticityWrapper(env._env, config=self.config['intrinsic_stochasticity']['action_dependent'])
            return env
        elif self.type == 2:
            raise NotImplementedError
        elif self.type == 3:
            raise NotImplementedError
        elif self.type == 4:
            raise NotImplementedError
        elif self.type == 5:
            return PartialObservationWrapper(env, config=self.config['partial_observation'])

if __name__ == "__main__":
    from atari import Atari

    env = Atari(
                'Breakout',
                4,
                [64, 64],
                gray=False,
                noops=0,
                lives='unused',
                sticky=False,
                actions='needed',
                resize='opencv',
                seed=0,
            )

    stochasticity_config = {
        'intrinsic_stochasticity': {
            'action_dependent': {
                'stochastic_action_prob': 0.5,
                },
            'action_independent': {
                'concept_drift': NotImplementedError
            },
        },
        'partial_observation': {
                'type': 'crop', # 'blackout' or 'crop' or 'ram'
                'mode': '1', # mode 1 in ram is buggy
                'prob': 0.75,
        },
    }

    stochasticity_wrapper = StochasticEnv(type=5, config=stochasticity_config)
    env = stochasticity_wrapper.get_env(env)
