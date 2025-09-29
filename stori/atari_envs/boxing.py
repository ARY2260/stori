import numpy as np
from stori.base_stochastic_env_classes import ActionDependentStochasticityWrapper, ActionIndependentRandomStochasticityWrapper, ActionIndependentConceptDriftWrapper, PartialObservationWrapper
from stori.utils import update_ram_state, blackout_obs, _save_restore_handler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


######################### RAM utils for Boxing #########################
Boxing_RAM_Mapping = {
                    # general states
                        "player_x": 32,
                        "player_y": 34,
                        "enemy_x": 33,
                        "enemy_y": 35,
                        "player_score": 18,
                        "enemy_score": 19,

                    # clock
                        "clock_minutes": 16,
                        "clock_seconds": 17,
                        "minutes_to_value": {0: np.uint8(11), 1: np.uint8(27)},
                        "seconds_to_value": {0: np.uint8(0),
                                              1: np.uint8(1),
                                              2: np.uint8(2),
                                              3: np.uint8(3),
                                              4: np.uint8(4), 
                                              5: np.uint8(5), 
                                              6: np.uint8(6), 
                                              7: np.uint8(7),
                                              8: np.uint8(8),
                                              9: np.uint8(9),
                                              10: np.uint8(16),
                                              11: np.uint8(17), 
                                              12: np.uint8(18), 
                                              13: np.uint8(19), 
                                              14: np.uint8(20), 
                                              15: np.uint8(21), 
                                              16: np.uint8(22), 
                                              17: np.uint8(23), 
                                              18: np.uint8(24), 
                                              19: np.uint8(25), 
                                              20: np.uint8(32), 
                                              21: np.uint8(33), 
                                              22: np.uint8(34), 
                                              23: np.uint8(35), 
                                              24: np.uint8(36), 
                                              25: np.uint8(37), 
                                              26: np.uint8(38), 
                                              27: np.uint8(39), 
                                              28: np.uint8(40), 
                                              29: np.uint8(41), 
                                              30: np.uint8(48), 
                                              31: np.uint8(49), 
                                              32: np.uint8(50), 
                                              33: np.uint8(51), 
                                              34: np.uint8(52), 
                                              35: np.uint8(53), 
                                              36: np.uint8(54), 
                                              37: np.uint8(55), 
                                              38: np.uint8(56), 
                                              39: np.uint8(57), 
                                              40: np.uint8(64), 
                                              41: np.uint8(65), 
                                              42: np.uint8(66), 
                                              43: np.uint8(67), 
                                              44: np.uint8(68), 
                                              45: np.uint8(69), 
                                              46: np.uint8(70), 
                                              47: np.uint8(71), 
                                              48: np.uint8(72), 
                                              49: np.uint8(73), 
                                              50: np.uint8(80), 
                                              51: np.uint8(81), 
                                              52: np.uint8(82), 
                                              53: np.uint8(83), 
                                              54: np.uint8(84), 
                                              55: np.uint8(85), 
                                              56: np.uint8(86), 
                                              57: np.uint8(87), 
                                              58: np.uint8(88), 
                                              59: np.uint8(89)},
                    # colors
                        "player_score_color": 1,
                        "enemy_score_color": 2,
                        "boxing_ring_color": 3,
                        "invisibility_background_color_value": 214,
                        "clock_color": 4,
                        "invisibility_clock_background_color_value": 208,
                    }

######################### Partial observation utils for Boxing #########################
FULL_BOXING_RING_REGION = {'min_x': np.int64(32),
                            'max_x': np.int64(127),
                            'min_y': np.int64(36),
                            'max_y': np.int64(176)}

LEFT_HALF_BOXING_RING_REGION = {'min_x': np.int64(32),
                            'max_x': np.int64(80),
                            'min_y': np.int64(36),
                            'max_y': np.int64(176)}

RIGHT_HALF_BOXING_RING_REGION = {'min_x': np.int64(81),
                            'max_x': np.int64(127),
                            'min_y': np.int64(36),
                            'max_y': np.int64(176)}

ENEMY_SCORE_REGION = {'min_x': np.int64(87),
                    'max_x': np.int64(134),
                    'min_y': np.int64(1),
                    'max_y': np.int64(16)}

PLAYER_SCORE_REGION = {'min_x': np.int64(33),
                    'max_x': np.int64(69),
                    'min_y': np.int64(1),
                    'max_y': np.int64(16)}

CLOCK_REGION = {'min_x': np.int64(57),
                'max_x': np.int64(97),
                'min_y': np.int64(14),
                'max_y': np.int64(27)}

BOXING_RING_BACKGROUND_COLOR = [110, 156, 66]

blackout_regions = [FULL_BOXING_RING_REGION, LEFT_HALF_BOXING_RING_REGION, RIGHT_HALF_BOXING_RING_REGION, ENEMY_SCORE_REGION, PLAYER_SCORE_REGION, CLOCK_REGION]

def blackout_obs_mode(array, mode='0'):
    """
    Black out regions of the observation according to the selected mode.

    Modes:
        '0': none - Do not black out any region
        '1': all - Black out all defined regions (boxing ring, enemy score, player score, clock)
        '2': left boxing ring - Black out the left boxing ring region
        '3': right boxing ring - Black out the right boxing ring region
        '4': full boxing ring - Black out the full boxing ring region
        '5': enemy score - Black out the enemy score region
        '6': player score - Black out the player score region
        '7': enemy score and player score - Black out the enemy score and player score region
        '8': clock - Black out the clock region
        '9': enemy score and player score and clock - Black out the enemy score, player score and clock region

    Args:
        array: numpy array of shape (height, width, 3)
        mode: str, one of ['1', '2', ..., '11', '0']

    Returns:
        array: numpy array with selected regions blacked out
    """
    if mode == '1':  # all
        for region in blackout_regions:
            array = blackout_obs(array, region, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '2':
        array = blackout_obs(array, LEFT_HALF_BOXING_RING_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '3':
        array = blackout_obs(array, RIGHT_HALF_BOXING_RING_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '4':
        array = blackout_obs(array, FULL_BOXING_RING_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '5':
        array = blackout_obs(array, ENEMY_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '6':
        array = blackout_obs(array, PLAYER_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '7':
        array = blackout_obs(array, ENEMY_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
        array = blackout_obs(array, PLAYER_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '8':
        array = blackout_obs(array, CLOCK_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '9':
        array = blackout_obs(array, ENEMY_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
        array = blackout_obs(array, PLAYER_SCORE_REGION, color=BOXING_RING_BACKGROUND_COLOR)
        array = blackout_obs(array, CLOCK_REGION, color=BOXING_RING_BACKGROUND_COLOR)
    elif mode == '0':  # none
        pass  # do nothing
    else:
        raise ValueError(f"Unknown blackout mode: {mode}")
    return array

######################### Ram modification utils for Boxing #########################
Boxing_Ring_update_ram_dict = {Boxing_RAM_Mapping['boxing_ring_color']: Boxing_RAM_Mapping['invisibility_background_color_value']}
Enemy_update_ram_dict = {Boxing_RAM_Mapping['enemy_score_color']: Boxing_RAM_Mapping['invisibility_background_color_value']}
Player_update_ram_dict = {Boxing_RAM_Mapping['player_score_color']: Boxing_RAM_Mapping['invisibility_background_color_value']}



def ram_obs_modification_mode(env, mode='0', get_obs=True, verbose=False):
    """
    Modify the RAM of the Breakout environment according to the selected mode.

    Modes:
        '0': none - Do not modify RAM
        '1': hide boxing ring
        '2': hide enemy
        '3': hide player

    Args:
        env: Boxing environment
        mode: str, one of ['0', '1', '2', '3']
        verbose: bool, whether to print debug info

    Returns:
        None (modifies env RAM in-place)
    """
    if mode == '0':
        return  # No modification

    elif mode == '1':  # hide boxing ring
        return _save_restore_handler(env, Boxing_Ring_update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '2':  # hide enemy
        return _save_restore_handler(env, Enemy_update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '3':  # hide player
        return _save_restore_handler(env, Player_update_ram_dict, get_obs=get_obs, verbose=verbose)
    else:
        raise ValueError(f"Unknown RAM modification mode: {mode}")


######################### Action independent random stochasticity wrapper for Boxing #########################
class BoxingActionIndependentRandomStochasticityWrapper(ActionIndependentRandomStochasticityWrapper):
   """Wrapper that implements action independent random stochasticity in internal ale _env.

   Note: Does not revert the score on screen.

   Modes:
      mode '0': No stochasticity.
      mode '1': Colorflip
      mode '2': Hit cancel
      mode '3': Displace to corners
   """

   def __init__(self, env, config):
      super().__init__(env, config)
      # logger.debug(f"setting original_ram_state and color_flipped to False in BoxingActionIndependentRandomStochasticityWrapper")
      self.original_ram_state = self.env.unwrapped.original_ram_state
      # logger.debug(f"original_ram_state: {self.original_ram_state}")
      self.color_flipped = False

   # def _timer_reset(self, action): # disabled because has uncertain affect on rewards
   #     """
   #     Takes affect if modified, in the next step.
   #     """
   #     obs, reward, done, info = self.env.step(action)
   #     if np.random.random() < self.prob:
   #         update_ram_dict = {16: self.original_ram_state[16], 17: self.original_ram_state[17]}
   #         update_ram_state(self.env, update_ram_dict)
   #     return obs, reward, done, info

   def _colorflip(self, action, verbose=False):
      """
      Flips the color of the enemy and player (character and score).
      """
      if np.random.random() < self.prob:
         self.color_flipped = not self.color_flipped
         if verbose:
               print("color flipped", self.color_flipped)
      if not self.color_flipped:
         update_ram_dict = {
                           Boxing_RAM_Mapping['player_score_color']: self.original_ram_state[Boxing_RAM_Mapping['player_score_color']],
                           Boxing_RAM_Mapping['enemy_score_color']: self.original_ram_state[Boxing_RAM_Mapping['enemy_score_color']],
                           }
         update_ram_state(self.env, update_ram_dict, verbose=False)
      else:
         update_ram_dict = {
                           Boxing_RAM_Mapping['player_score_color']: self.original_ram_state[Boxing_RAM_Mapping['enemy_score_color']],
                           Boxing_RAM_Mapping['enemy_score_color']: self.original_ram_state[Boxing_RAM_Mapping['player_score_color']],
                           }
         update_ram_state(self.env, update_ram_dict, verbose=False)
      return self.env.step(action)

   def _hit_cancel(self, action, verbose=False):
      """
      Reverts the score to the previous state if player or enemy is hit.
      """
      self.previous_ram_state = self.env.unwrapped.ale.getRAM()
      obs, reward, done, truncated, info = self.env.step(action)
      current_ram_state = self.env.unwrapped.ale.getRAM()
      changed_indices = [i for i in [Boxing_RAM_Mapping['player_score'], Boxing_RAM_Mapping['enemy_score']] if current_ram_state[i] != self.previous_ram_state[i]]
      if len(changed_indices) == 1 and np.random.random() < self.prob:
         self.env.unwrapped.ale.setRAM(changed_indices[0], self.previous_ram_state[changed_indices[0]])
         if verbose:
               if changed_indices[0] == Boxing_RAM_Mapping['player_score']:
                  print(f"player hit cancelled")
               else:
                  print(f"enemy hit cancelled")
         print(f"reward: {reward}")
         reward = 0.0
         if verbose:
               print(f"reward reverted to 0")
      return obs, reward, done, truncated, info


   def _displace_to_corners(self, action, verbose=False):
      """
      Displaces the player and enemy to the corners of the screen.
      """
      if np.random.random() < self.prob:
         self.env.unwrapped.ale.setRAM(Boxing_RAM_Mapping['player_x'], self.original_ram_state[Boxing_RAM_Mapping['enemy_x']])
         self.env.unwrapped.ale.setRAM(Boxing_RAM_Mapping['player_y'], self.original_ram_state[Boxing_RAM_Mapping['enemy_y']])
         self.env.unwrapped.ale.setRAM(Boxing_RAM_Mapping['enemy_x'], self.original_ram_state[Boxing_RAM_Mapping['player_x']])
         self.env.unwrapped.ale.setRAM(Boxing_RAM_Mapping['enemy_y'], self.original_ram_state[Boxing_RAM_Mapping['player_y']])
         if verbose:
               print(f"player and enemy displaced to corners")
      return self.env.step(action)


   def reset(self, seed=None, options=None):
      logger.debug(f"using reset() method of BoxingActionIndependentRandomStochasticityWrapper")
      obs = self.env.reset(seed=seed, options=options)
      self.original_ram_state = self.env.unwrapped.ale.getRAM()
      self.color_flipped = False
      return obs

   def step(self, action):
      if self.mode == '0':
         return self.env.step(action)
      elif self.mode == '1': # keep prod around 0.001
         return self._colorflip(action, verbose=True)
      elif self.mode == '2': # keep prod around 0.25
         return self._hit_cancel(action, verbose=True)
      elif self.mode == '3': # keep prod around 0.001
         return self._displace_to_corners(action, verbose=True)
      else:
         raise NotImplementedError

######################### Partial observation wrapper for Boxing #########################
class BoxingPartialObservationWrapper(PartialObservationWrapper):
   def _blackout_obs_mode(self, array, mode) -> None:
      blackout_obs_mode(array, mode)

   def _ram_obs_mode(self, env, mode) -> np.ndarray:
      return ram_obs_modification_mode(env, mode, get_obs=True, verbose=False)

######################### Wrapper registry for Boxing #########################

def get_boxing_wrapper_registry():
   return {
      'action_dependent': ActionDependentStochasticityWrapper,
      'action_independent_random': BoxingActionIndependentRandomStochasticityWrapper,
      'action_independent_concept_drift': ActionIndependentConceptDriftWrapper,
      'partial_observation': BoxingPartialObservationWrapper,
   }
