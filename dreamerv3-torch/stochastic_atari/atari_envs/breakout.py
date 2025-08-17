import random
import numpy as np
from stochastic_atari.base_stochastic_env_classes import ActionDependentStochasticityWrapper, ActionIndependentRandomStochasticityWrapper, ActionIndependentConceptDriftWrapper, PartialObservationWrapper
from stochastic_atari.utils import update_ram_state, blackout_obs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


######################### Blackout Partial observation utils for Breakout #########################
Blocks_region = {'min_x': 8,
                'max_x': 151,
                'min_y': 57,
                'max_y': 92}

Paddle_region = {'min_x': 7,
                'max_x': 152,
                'min_y': 189,
                'max_y': 195}

Score_region = {'min_x': 2,
                'max_x': 158,
                'min_y': 3,
                'max_y': 16}

Ball_missing_region = {'top': {'min_x': 8,
                                'max_x': 151,
                                'min_y': 105,
                                'max_y': 130},
                        'middle': {'min_x': 8,
                                    'max_x': 151,
                                    'min_y': 131,
                                    'max_y': 156},
                        'bottom': {'min_x': 8,
                                    'max_x': 151,
                                    'min_y': 157,
                                    'max_y': 182}
                    }

blackout_regions = [Blocks_region, Paddle_region, Score_region, Ball_missing_region['top'], Ball_missing_region['middle'], Ball_missing_region['bottom']]



def blackout_obs_mode(array, mode='0'):
    """
    Black out regions of the observation according to the selected mode.

    Modes:
        '0': none - Do not black out any region
        '1': all - Black out all defined regions (blocks, paddle, score, ball missing regions)
        '2': blocks - Black out only the blocks region
        '3': paddle - Black out only the paddle region
        '4': score - Black out only the score region
        '5': ball_missing_top - Black out only the top ball missing region
        '6': ball_missing_middle - Black out only the middle ball missing region
        '7': ball_missing_bottom - Black out only the bottom ball missing region
        '8': blocks_and_paddle - Black out both blocks and paddle regions
        '9': blocks_and_score - Black out both blocks and score regions
        '10': ball_missing_top_and_bottom - Black out both top and bottom ball missing regions
        '11': ball_missing_all - Black out all ball missing regions (top, middle, bottom)

    Args:
        obs: numpy array of shape (height, width, 3)
        mode: str, one of ['1', '2', ..., '11', '0']

    Returns:
        modified_obs: numpy array with selected regions blacked out
    """
    if mode == '1':  # all
        for region in blackout_regions:
            array = blackout_obs(array, region)
    elif mode == '2':  # blocks
        array = blackout_obs(array, Blocks_region)
    elif mode == '3':  # paddle
        array = blackout_obs(array, Paddle_region)
    elif mode == '4':  # score
        array = blackout_obs(array, Score_region)
    elif mode == '5':  # ball_missing_top
        array = blackout_obs(array, Ball_missing_region['top'])
    elif mode == '6':  # ball_missing_middle
        array = blackout_obs(array, Ball_missing_region['middle'])
    elif mode == '7':  # ball_missing_bottom
        array = blackout_obs(array, Ball_missing_region['bottom'])
    elif mode == '8':  # blocks_and_paddle
        array = blackout_obs(array, Blocks_region)
        array = blackout_obs(array, Paddle_region)
    elif mode == '9':  # blocks_and_score
        array = blackout_obs(array, Blocks_region)
        array = blackout_obs(array, Score_region)
    elif mode == '10':  # ball_missing_top_and_bottom
        array = blackout_obs(array, Ball_missing_region['top'])
        array = blackout_obs(array, Ball_missing_region['bottom'])
    elif mode == '11':  # ball_missing_all
        array = blackout_obs(array, Ball_missing_region['top'])
        array = blackout_obs(array, Ball_missing_region['middle'])
        array = blackout_obs(array, Ball_missing_region['bottom'])
    elif mode == '0':  # none
        pass  # do nothing
    else:
        raise ValueError(f"Unknown blackout mode: {mode}")
    return array

######################### RAM observation modification utils for Breakout #########################
Breakout_RAM_Mapping = {
    # general states
    "ball_x":99,
    "ball_y":101,
    "player_x":72,
    "blocks_hit_count":77,
    "score":84,

    # blocks
    "block_bit_map": [
        {   "column": 0, # right to left
            "indices": [0, 1, 2, 3, 4, 5], # bottom to top
            "bit_size": 6,
            "fill_direction": 1,
        },
        {
            "column": 1,  # right to left
            "indices": [6, 7, 8, 9, 10, 11], # bottom to top
            "bit_size": 8,
            "fill_direction": -1,
        },
        {
            "column": 2,  # right to left
            "indices": [12, 13, 14, 15, 16, 17], # bottom to top
            "bit_size": 4,
            "fill_direction": 1,
        },
        {
            "column": 3,  # right to left
            "indices": [18, 19, 20, 21, 22, 23], # bottom to top
            "bit_size": 8,
            "fill_direction": 1,
        },
        {
            "column": 4,  # right to left
            "indices": [24, 25, 26, 27, 28, 29], # bottom to top
            "bit_size": 8,
            "fill_direction": -1,
        },
        {
            "column": 5,  # right to left
            "indices": [30, 31, 32, 33, 34, 35], # bottom to top
            "bit_size": 2,
            "fill_direction": 1,
        },
    ],

    # color states
    "ball_paddle_color": 62,
    "blocks_row_1_color": 64,
    "blocks_row_2_color": 65,
    "blocks_row_3_color": 66,
    "blocks_row_4_color": 67,
    "blocks_row_5_color": 68,
    "blocks_row_6_color": 69,

    # scores visibility
    "digit_1": 81,
    "digit_2": 83,
    "digit_3": 85,
}

NUS_pattern_blocks_ram_mapping = {0: 62,
 1: 61,
 2: 60,
 3: 62,
 5: 60,
 6: 240,
 7: 127,
 8: 120,
 9: 112,
 10: 119,
 11: 120,
 12: 0,
 13: 240,
 14: 240,
 15: 240,
 16: 240,
 17: 240,
 18: 57,
 19: 218,
 20: 219,
 21: 219,
 22: 219,
 23: 219,
 24: 239,
 25: 239,
 26: 238,
 27: 237,
 28: 235,
 29: 231}

Ball_hidden_ram_mapping = {
    Breakout_RAM_Mapping['ball_x']: 0,
    Breakout_RAM_Mapping['ball_y']: 0,
}

def ram_obs_modification_mode(env, mode='0', verbose=False):
    """
    Modify the RAM of the Breakout environment according to the selected mode.

    Modes:
        '0': none - Do not modify RAM
        '1': nus_pattern - Apply NUS pattern to blocks RAM
        '2': ball_hidden - Hide the ball by setting its RAM values to 0

    Args:
        env: Breakout environment
        mode: str, one of ['0', '1', '2']
        verbose: bool, whether to print debug info

    Returns:
        None (modifies env RAM in-place)
    """
    if mode == '0':
        return  # No modification

    elif mode == '1':  # nus_pattern (buggy)
        # save env state before modification
        saved_state = env.unwrapped.clone_state(True)
        update_ram_state(env, NUS_pattern_blocks_ram_mapping, verbose=verbose)
        obs, _, _, _ = env.step(0)
        # restore env state
        env.unwrapped.restore_state(saved_state)
        return obs

    elif mode == '2':  # ball_hidden
        # save env state before modification
        saved_state = env.unwrapped.clone_state(True)
        update_ram_state(env, Ball_hidden_ram_mapping, verbose=verbose)
        obs, _, _, _ = env.step(0)
        # restore env state
        env.unwrapped.restore_state(saved_state)
        return obs

    else:
        raise ValueError(f"Unknown RAM modification mode: {mode}")


######################### Action independent random stochasticity wrapper for Breakout #########################
class BreakoutActionIndependentRandomStochasticityWrapper(ActionIndependentRandomStochasticityWrapper):
    """Wrapper that implements action independent random stochasticity in internal ale _env.

    Note: Does not revert the score on screen.

    Modes:
       mode '0': No stochasticity.
       mode '1': Block hit cancel - if a block is hit, the RAM is reverted to the previous state, effectively canceling the block hit.
       mode '2': Block hit cancel (reward reverted) - if a block is hit, the RAM is reverted to the previous state, effectively canceling the block hit.
       mode '3': Regenerate hit block - after a block is hit, with some probability, a randomly picked block RAM index is reverted to its original value, regenerating a block.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.original_ram_state = self.env.unwrapped.original_ram_state

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


######################### Partial observation wrapper for Breakout #########################
class BreakoutPartialObservationWrapper(PartialObservationWrapper):
    def _blackout_obs_mode(self, array, mode) -> None:
        blackout_obs_mode(array, mode=mode)

    def _ram_obs_mode(self, env, mode) -> np.ndarray:
        return ram_obs_modification_mode(env, mode=mode, verbose=False)

######################### Wrapper registry for Breakout #########################

def get_breakout_wrapper_registry():
    return {
        'action_dependent': ActionDependentStochasticityWrapper,
        'action_independent_random': BreakoutActionIndependentRandomStochasticityWrapper,
        'action_independent_concept_drift': ActionIndependentConceptDriftWrapper,
        'partial_observation': BreakoutPartialObservationWrapper,
    }
