import numpy as np
import random
from post_processing_obs import blackout_obs_mode, crop_obs_mode
from ram_modification_obs import ram_obs_modification_mode
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


class ActionIndependentRandomStochasticityWrapper(ActionWrapper):
    """Wrapper that implements action independent random stochasticity in internal ale _env.

    Note: Does not revert the score on screen.

    Modes:
       mode '0': No stochasticity.
       mode '1': Block hit cancel - if a block is hit, the RAM is reverted to the previous state, effectively canceling the block hit.
       mode '2': Block hit cancel (reward reverted) - if a block is hit, the RAM is reverted to the previous state, effectively canceling the block hit.
       mode '3': Regenerate hit block - after a block is hit, with some probability, a randomly picked block RAM index is reverted to its original value, regenerating a block.
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.mode = config['mode']
        self.prob = config['random_stochasticity_prob']

    def _block_hit_cancel(self, action, cancel_reward=True, verbose=False):
        """
        Takes affect if modified, in the next step.
        If cancel_reward is True, the reward is set to 0 if block hit is cancelled.
        """
        self.previous_ram_state = self.env.unwrapped.ale.getRAM()
        obs, reward, done, info = self.env.step(action)
        current_ram_state = self.env.unwrapped.ale.getRAM()
        if any(current_ram_state[:36] != self.original_ram_state[:36]):
            changed_indices = [i for i in range(36) if current_ram_state[i] != self.previous_ram_state[i]]
            if len(changed_indices) > 0 and random.random() < self.prob:
                for i in changed_indices:
                    self.env.unwrapped.ale.setRAM(i, self.previous_ram_state[i])
                ## reset scores and lives to previous state (does not work)
                # for i in range(81,91):
                #     if verbose:
                #         print(f"resetting {i} to {self.previous_ram_state[i]} from {current_ram_state[i]}")
                #     self.env.unwrapped.ale.setRAM(i, self.previous_ram_state[i])
                if verbose:
                    print(f"block hit cancelled")
                if cancel_reward:
                    reward = 0.0
                    if verbose:
                        print(f"reward reverted to 0")
        return obs, reward, done, info

    def _regenerate_hit_block(self, action, verbose=False):
        """
        Takes affect if modified, in the next step.
        """
        obs, reward, done, info = self.env.step(action)
        current_ram_state = self.env.unwrapped.ale.getRAM()
        # list of indices that changed
        changed_indices = [i for i in range(36) if current_ram_state[i] != self.original_ram_state[i]]
        # randomly choose a changed index and update its value to the original value
        if len(changed_indices) > 0 and random.random() < self.prob:
            random_index = random.choice(changed_indices)
            self.env.unwrapped.ale.setRAM(random_index, self.original_ram_state[random_index])
            if verbose:
                print(f"hit block regenerated")
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self.original_ram_state = self.env.unwrapped.ale.getRAM()
        return obs

    def step(self, action):
        if self.mode == '0':
            return self.env.step(action)
        elif self.mode == '1':
            return self._block_hit_cancel(action, cancel_reward=False, verbose=True)
        elif self.mode == '2':
            return self._block_hit_cancel(action, cancel_reward=True, verbose=True)
        elif self.mode == '3':
            return self._regenerate_hit_block(action)


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
            env._env = ActionIndependentRandomStochasticityWrapper(env._env, config=self.config['intrinsic_stochasticity']['action_independent_random'])
            return env
        elif self.type == 3:
            raise NotImplementedError
        elif self.type == 4: # default
            raise env
        elif self.type == 5:
            return PartialObservationWrapper(env, config=self.config['partial_observation'])

if __name__ == "__main__":
    from atari_dreamerv3 import Atari

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
            'action_independent_random': {
                'mode': '2',
                'random_stochasticity_prob': 0.25, # mode 3: keep around 0.0005
            },
            'action_independent_concept_drift': {
                'concept_drift': NotImplementedError
            },
        },
        'partial_observation': {
                'type': 'ram', # 'blackout' or 'crop' or 'ram'
                'mode': '2', # mode 1 in ram is buggy
                'prob': 0.5,
        },
    }

    stochasticity_wrapper = StochasticEnv(type=5, config=stochasticity_config)
    env = stochasticity_wrapper.get_env(env)
