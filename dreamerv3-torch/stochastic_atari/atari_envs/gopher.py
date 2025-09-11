import numpy as np
from stochastic_atari.base_stochastic_env_classes import ActionDependentStochasticityWrapper, ActionIndependentRandomStochasticityWrapper, ActionIndependentConceptDriftWrapper, PartialObservationWrapper
from stochastic_atari.utils import update_ram_state, blackout_obs, _save_restore_handler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######################### RAM utils for Gopher #########################
Gopher_RAM_Mapping = {
    # general states
    'player_x_pos': 31, # [19 (min), 83 (mid), 148 (max)]
    'score': [48, 49, 50],

    # underground (row top to bottom)
    'underground': {
        'row_0': [0, 1, 2, 3, 4, 5],
        'row_1': [6, 7, 8, 9, 10, 11],
        'row_2': [12, 13, 14, 15, 16, 17],
        'row_3': [18, 19, 20, 21, 22, 23],
    },

    # carrots visibility
    'carrot_visibility': 52,
    'carrot_visibility_states': {
        '000': {52: 0},
        '001': {52: 1},
        '010': {52: 2},  
        '011': {52: 3},
        '100': {52: 4},
        '101': {52: 5},
        '110': {52: 6},
        '111': {52: 7},
    },  

    # duck
    'duck_image': {
        25: 254,
        26: 201,
        27: 254,
    },
    'duck_x_pos': 28, # [10, 146]
    'duck_state': 94,
    'duck_state_settings': {
        'hidden': {94: 0},
        'flap_middle': {94: 1},
        'flap_up': {94: 9},
        'flap_down': {94: 25},
    },

    # seed
    'seed_y': 92, # [1, 108] (hide at 128)
    'seed_x': 95, # [4, 168]

    'delayed_start_step': 238,
}

######################### Partial observation utils for Gopher #########################

GOPHER_ATTACK_REGION = {
    'LEFT': {
        'ROW_0': {'min_x': np.int64(0),
                    'max_x': np.int64(59),
                    'min_y': np.int64(146),
                    'max_y': np.int64(150)},
        'ROW_1': {'min_x': np.int64(0),
                    'max_x': np.int64(59),
                    'min_y': np.int64(151),
                    'max_y': np.int64(155)},
        'ROW_2': {'min_x': np.int64(0),
                    'max_x': np.int64(59),
                    'min_y': np.int64(156),
                    'max_y': np.int64(160)},
        },
    'RIGHT': {
        'ROW_0': {'min_x': np.int64(100),
                    'max_x': np.int64(159),
                    'min_y': np.int64(146),
                    'max_y': np.int64(150)},
        'ROW_1': {'min_x': np.int64(100),
                    'max_x': np.int64(159),
                    'min_y': np.int64(151),
                    'max_y': np.int64(155)},
        'ROW_2': {'min_x': np.int64(100),
                    'max_x': np.int64(159),
                    'min_y': np.int64(156),
                    'max_y': np.int64(160)},
    },
    'COLORS': {
        'ROW_0': [101,209,174],
        'ROW_1': [117,231,194],
        'ROW_2': [132,252,212],
    },
}

UNDER_GROUND_FULL_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(161),
                            'max_y': np.int64(200)}

UNDER_GROUND_FULL_OFFSET_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(164),
                            'max_y': np.int64(200)}

UNDER_GROUND_ROW_0_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(161),
                            'max_y': np.int64(168)}

UNDER_GROUND_ROW_1_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(168),
                            'max_y': np.int64(176)}

UNDER_GROUND_ROW_2_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(176),
                            'max_y': np.int64(182)}

UNDER_GROUND_ROW_3_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(183),
                            'max_y': np.int64(194)}

UNDERGROUND_COLORS = {
    'BEFORE_DUG_UNDERGROUND': [187,159,71],
    'AFTER_DUG_UNDERGROUND': [223,183,85],
}

FARMER_BELOW_NOSE_REGION = {'min_x': np.int64(0),
                            'max_x': np.int64(159),
                            'min_y': np.int64(104),
                            'max_y': np.int64(145)}

FARMER_FULL_REGION = {'min_x': np.int64(0),
                        'max_x': np.int64(159),
                        'min_y': np.int64(87),
                        'max_y': np.int64(145)}

DUCK_FLY_REGION = {'min_x': np.int64(0),
                    'max_x': np.int64(159),
                    'min_y': np.int64(26),
                    'max_y': np.int64(56)}

SCORE_REGION = {'min_x': np.int64(0),
                'max_x': np.int64(159),
                'min_y': np.int64(0),
                'max_y': np.int64(25)}

blackout_regions = [GOPHER_ATTACK_REGION, UNDER_GROUND_FULL_REGION, UNDER_GROUND_FULL_OFFSET_REGION, UNDER_GROUND_ROW_0_REGION, UNDER_GROUND_ROW_1_REGION, UNDER_GROUND_ROW_2_REGION, UNDER_GROUND_ROW_3_REGION, FARMER_BELOW_NOSE_REGION, FARMER_FULL_REGION, DUCK_FLY_REGION, SCORE_REGION]

def blackout_gopher_attack_region(array, attack_regions, left=True, right=True):
    """
    Black out the gopher attack region of the observation.
    """
    if left:
        array = blackout_obs(array, attack_regions['LEFT']['ROW_0'], color=attack_regions['COLORS']['ROW_0'])
        array = blackout_obs(array, attack_regions['LEFT']['ROW_1'], color=attack_regions['COLORS']['ROW_1'])
        array = blackout_obs(array, attack_regions['LEFT']['ROW_2'], color=attack_regions['COLORS']['ROW_2'])
    if right:
        array = blackout_obs(array, attack_regions['RIGHT']['ROW_0'], color=attack_regions['COLORS']['ROW_0'])
        array = blackout_obs(array, attack_regions['RIGHT']['ROW_1'], color=attack_regions['COLORS']['ROW_1'])
        array = blackout_obs(array, attack_regions['RIGHT']['ROW_2'], color=attack_regions['COLORS']['ROW_2'])
    return array

def blackout_obs_mode(array, mode='0'):
    """
    Black out regions of the observation according to the selected mode.

    Modes:
        '0': none - Do not black out any region
        '1': all - Black out all defined regions (gopher attack, underground, farmer, duck fly, score)
        '2': gopher attack - Black out both left and right gopher attack regions
        '3': left gopher attack - Black out only the left gopher attack region
        '4': right gopher attack - Black out only the right gopher attack region
        '5': underground full - Black out the full underground region (without offset, using default 'before dug' color)
        '6': underground full offset - Black out the full underground region with a slight vertical offset (using default 'before dug' color); this mimics the real world where the farmer can see slight holes or disturbances in the ground, providing subtle cues
        '7': underground row 0 - Black out only underground row 0 region (using default 'before dug' color)
        '8': underground row 0 (dug color) - Black out only underground row 0 region (using 'after dug' color, to confuse agent about whether it is dug)
        '9': underground row 1 - Black out only underground row 1 region (using default 'before dug' color)
        '10': underground row 1 (dug color) - Black out only underground row 1 region (using 'after dug' color, to confuse agent about whether it is dug)
        '11': underground row 2 - Black out only underground row 2 region (using default 'before dug' color)
        '12': underground row 2 (dug color) - Black out only underground row 2 region (using 'after dug' color, to confuse agent about whether it is dug)
        '13': underground row 3 - Black out only underground row 3 region (using default 'before dug' color)
        '14': underground row 3 (dug color) - Black out only underground row 3 region (using 'after dug' color, to confuse agent about whether it is dug)
        '15': farmer - Black out all farmer regions
        '16': farmer below nose - Black out only the region below the farmer's nose (so the agent can still locate the farmer, but cannot know what the farmer is doing)
        '17': duck fly - Black out the duck fly region
        '18': score - Black out the score region

    Args:
        array: numpy array of shape (height, width, 3)
        mode: str, one of ['0', '1', ..., '18']

    Returns:
        array: numpy array with selected regions blacked out
    """
    if mode == '0':  # none
        return array
    elif mode == '1':  # all
        # Black out all regions, but handle gopher attack regions with their special colors
        # First, blackout all regions except gopher attack
        for region in blackout_regions:
            if region not in [GOPHER_ATTACK_REGION]:
                # For underground regions, use UNDERGROUND_COLORS where appropriate
                if region in [UNDER_GROUND_FULL_REGION, UNDER_GROUND_FULL_OFFSET_REGION,
                              UNDER_GROUND_ROW_0_REGION, UNDER_GROUND_ROW_1_REGION,
                              UNDER_GROUND_ROW_2_REGION, UNDER_GROUND_ROW_3_REGION]:
                    array = blackout_obs(array, region, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
                else:
                    array = blackout_obs(array, region)
        # Now blackout gopher attack regions with their row-specific colors
        array = blackout_gopher_attack_region(array, GOPHER_ATTACK_REGION, left=True, right=True)
        return array
    elif mode == '2':  # gopher attack (both sides)
        array = blackout_gopher_attack_region(array, GOPHER_ATTACK_REGION, left=True, right=True)
        return array
    elif mode == '3':  # left gopher attack
        array = blackout_gopher_attack_region(array, GOPHER_ATTACK_REGION, left=True, right=False)
        return array
    elif mode == '4':  # right gopher attack
        array = blackout_gopher_attack_region(array, GOPHER_ATTACK_REGION, left=False, right=True)
        return array
    elif mode == '5':  # underground full (before dug color)
        array = blackout_obs(array, UNDER_GROUND_FULL_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '6':  # underground full offset (before dug color)
        array = blackout_obs(array, UNDER_GROUND_FULL_OFFSET_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '7':  # underground row 0 (before dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_0_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '8':  # underground row 0 (dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_0_REGION, color=UNDERGROUND_COLORS['AFTER_DUG_UNDERGROUND'])
        return array
    elif mode == '9':  # underground row 1 (before dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_1_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '10':  # underground row 1 (dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_1_REGION, color=UNDERGROUND_COLORS['AFTER_DUG_UNDERGROUND'])
        return array
    elif mode == '11':  # underground row 2 (before dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_2_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '12':  # underground row 2 (dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_2_REGION, color=UNDERGROUND_COLORS['AFTER_DUG_UNDERGROUND'])
        return array
    elif mode == '13':  # underground row 3 (before dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_3_REGION, color=UNDERGROUND_COLORS['BEFORE_DUG_UNDERGROUND'])
        return array
    elif mode == '14':  # underground row 3 (dug color)
        array = blackout_obs(array, UNDER_GROUND_ROW_3_REGION, color=UNDERGROUND_COLORS['AFTER_DUG_UNDERGROUND'])
        return array
    elif mode == '15':  # farmer (full)
        array = blackout_obs(array, FARMER_FULL_REGION)
        return array
    elif mode == '16':  # farmer below nose
        array = blackout_obs(array, FARMER_BELOW_NOSE_REGION)
        return array
    elif mode == '17':  # duck fly
        array = blackout_obs(array, DUCK_FLY_REGION)
        return array
    elif mode == '18':  # score
        array = blackout_obs(array, SCORE_REGION)
        return array
    else:
        raise ValueError(f"Unknown blackout mode: {mode}")

######################### Ram modification utils for Gopher #########################

CARROT_VIS_ADDR = Gopher_RAM_Mapping['carrot_visibility']
CARROT_VIS_STATES = Gopher_RAM_Mapping['carrot_visibility_states']
SEED_Y_ADDR = Gopher_RAM_Mapping['seed_y']

def ram_obs_modification_mode(env, mode='0', get_obs=True, verbose=False):
    """
    Modify the RAM of the Gopher environment according to the selected mode.

    Modes:
        '0': none - Do not modify RAM.
        '1': hide left carrot - Hide the left carrot in RAM.
        '2': hide middle carrot - Hide the middle carrot in RAM.
        '3': hide right carrot - Hide the right carrot in RAM.
        '4': hide all carrots - Hide all carrots in RAM.
        '5': hide seed - Hide the seed in RAM.

    Args:
        env: Gopher environment.
        mode: str, one of ['0', '1', '2', '3', '4', '5'].
        get_obs: bool, whether to return the observation after modification.
        verbose: bool, whether to print debug info.

    Returns:
        None or modified observation, depending on get_obs.
    """

    def _carrot_update_dict(env, left=None, middle=None, right=None):
        """
        left, middle, right: 1=visible, 0=hidden, or None to leave unchanged.
        Only hides carrots that are currently visible; does not unhide hidden carrots.
        """
        ram = env.unwrapped.ale.getRAM()
        val = ram[CARROT_VIS_ADDR]
        # Extract current carrot bits: left (bit 2), middle (bit 1), right (bit 0)
        bits = [
            (val >> 2) & 1,  # left
            (val >> 1) & 1,  # middle
            val & 1          # right
        ]
        # Only set to 0 if requested and currently visible; never set to 1 if already hidden
        if left is not None and left == 0:
            bits[0] = 0
        if middle is not None and middle == 0:
            bits[1] = 0
        if right is not None and right == 0:
            bits[2] = 0
        # Never set any bit to 1 (unhide) if it is already 0
        key = f"{bits[0]}{bits[1]}{bits[2]}"
        return CARROT_VIS_STATES[key]

    if mode == '0':
        return  # No modification

    elif mode == '1':  # hide left carrot
        update_ram_dict = _carrot_update_dict(env, left=0)
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '2':  # hide middle carrot
        update_ram_dict = _carrot_update_dict(env, middle=0)
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '3':  # hide right carrot
        update_ram_dict = _carrot_update_dict(env, right=0)
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '4':  # hide all carrots
        # Only hide carrots that are currently visible
        update_ram_dict = _carrot_update_dict(env, left=0, middle=0, right=0)
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '5':  # hide seed
        # Set seed_y to 128 (hidden)
        update_ram_dict = {SEED_Y_ADDR: 128}
        # Only update if not already hidden
        ram = env.unwrapped.ale.getRAM()
        if ram[SEED_Y_ADDR] == 128:
            return None  # Already hidden, nothing to do
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    else:
        raise ValueError(f"Unknown RAM modification mode: {mode}")

######################### Action independent random stochasticity wrapper for Gopher #########################

class GopherActionIndependentRandomStochasticityWrapper(ActionIndependentRandomStochasticityWrapper):
    """
    Wrapper that implements action independent random stochasticity in internal ale _env for Gopher.

    Modes:
        mode '0': No stochasticity.
        mode '1': Hole doesn't close (cancellation of filling of holes underground).
        mode '2': Randomly remove (kill) one of the carrots.
    """

    def __init__(self, env, config):
      super().__init__(env, config)
      self.original_ram_state = self.env.unwrapped.original_ram_state
      self._carrot_removal_done = False
    
    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self.original_ram_state = self.env.unwrapped.ale.getRAM()
        self._carrot_removal_done = False
        return obs

    def _hole_filling_cancel(self, action, cancel_reward=True, verbose=False):
        """
        Takes affect if modified, in the next step.
        If cancel_reward is True, the reward is set to 0 if hole filling is cancelled.
        """
        self.previous_ram_state = self.env.unwrapped.ale.getRAM()
        obs, reward, done, info = self.env.step(action)
        
        current_ram_state = self.env.unwrapped.ale.getRAM()
        if action in [1, 5, 6, 7]: # only for digging actions
            underground_indices = [i for row in Gopher_RAM_Mapping['underground'].values() for i in row]
            if any(current_ram_state[underground_indices] != self.original_ram_state[underground_indices]):
                changed_indices = [i for i in underground_indices if current_ram_state[i] != self.previous_ram_state[i]]
                # only if the changed index overlaps with the player's x position
                confirm_overlap_indices = []
                if len(changed_indices) > 0 and np.random.random() < self.prob:
                    # INSERT_YOUR_CODE
                    # Map underground indices to their corresponding x position ranges
                    # Each underground index corresponds to a column; columns are 6 wide, starting at x=19
                    player_x_addr = Gopher_RAM_Mapping['player_x_pos']
                    player_x = current_ram_state[player_x_addr]
                    if verbose:
                        print(f"player x: {player_x}")
                    # There are 24 underground indices, 6 columns, 4 rows
                    # Columns: 0-5, x ranges: 19+col*22 to 19+(col+1)*22-1 (approx, since 148-19=129, 129/6=21.5)
                    col_width = (148 - 19 + 1) / 6  # 21
                    col_starts = [19 + col_width * c for c in range(6)]
                    col_ends = [start + col_width - 1 for start in col_starts]
                    if verbose:
                        print(f"col starts: {col_starts}")
                        print(f"col ends: {col_ends}")
                        print(f"changed indices: {changed_indices}")
                    for i in changed_indices:
                        # Find which column this index is in
                        col = i % 6
                        x_start = col_starts[col]
                        x_end = col_ends[col]
                        if x_start <= player_x <= x_end:
                            if verbose:
                                print(f"Changed underground index {i} (col {col}, x range {x_start}-{x_end}) overlaps with player at x={player_x}")
                            confirm_overlap_indices.append(i)
                    if len(confirm_overlap_indices) > 0:
                        for i in confirm_overlap_indices:
                            self.env.unwrapped.ale.setRAM(i, self.previous_ram_state[i])
                        if verbose:
                            print(f"hole filling cancelled")
                        if cancel_reward:
                            print("initial reward: ", reward)
                            reward = 0.0
                            if verbose:
                                print(f"reward reverted to 0")
                    else:
                        if verbose:
                            print(f"No changed underground index overlaps with player")
        return obs, reward, done, info

    def _random_remove_carrot(self, action, verbose=False, is_delayed_start=False):
        """
        Randomly remove (kill) one of the carrots.
        Only do this once per reset.
        """
        obs, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        if not self._carrot_removal_done and not is_delayed_start:
            carrot_vis_addr = Gopher_RAM_Mapping['carrot_visibility']
            visible = [(ram[carrot_vis_addr] >> i) & 1 for i in range(3)]
            if len(visible) < 2: # at least 2 carrots must be visible
                self._carrot_removal_done = True
                if verbose:
                    print(f"At least 2 carrots must be visible, so no carrot removal")
                return obs, reward, done, info
            indices = [i for i, v in enumerate(visible) if v]
            if indices and np.random.random() < self.prob:
                idx = np.random.choice(indices)
                new_val = ram[carrot_vis_addr] & ~(1 << idx)
                update_ram_state(self.env, {carrot_vis_addr: new_val})
                if verbose:
                    print(f"A carrot died!")
                self._carrot_removal_done = True
        return obs, reward, done, info

    def step(self, action):
        """
        Implements the action-independent random stochasticity for Gopher.

        Modes:
            mode '0': No stochasticity.
            mode '1': Hole doesn't close (cancellation of filling of holes underground).
            mode '2': Hole doesn't close with reward reverted to 0 (cancellation of filling of holes underground).
            mode '3': Randomly remove (kill) one of the carrots (only once per reset).
        """
        if self.mode == '0':
            return self.env.step(action)
        elif self.mode == '1':
            return self._hole_filling_cancel(action, cancel_reward=False, verbose=True)
        elif self.mode == '2':
            return self._hole_filling_cancel(action, cancel_reward=True, verbose=True)
        elif self.mode == '3':
            is_delayed_start=self.env.unwrapped.ale.getFrameNumber() < Gopher_RAM_Mapping['delayed_start_step']
            return self._random_remove_carrot(action, verbose=True, is_delayed_start=is_delayed_start)
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")


######################### Partial observation wrapper for Gopher #########################
class GopherPartialObservationWrapper(PartialObservationWrapper):
    def _blackout_obs_mode(self, array, mode) -> None:
        blackout_obs_mode(array, mode)
    
    def _ram_obs_mode(self, env, mode) -> np.ndarray:
        return ram_obs_modification_mode(env, mode, get_obs=True, verbose=False)

######################### Wrapper registry for Gopher #########################

def get_gopher_wrapper_registry():
    return {
        'action_dependent': ActionDependentStochasticityWrapper,
        'action_independent_random': GopherActionIndependentRandomStochasticityWrapper,
        'action_independent_concept_drift': ActionIndependentConceptDriftWrapper,
        'partial_observation': GopherPartialObservationWrapper,
    }
