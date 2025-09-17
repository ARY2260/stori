import numpy as np
from stochastic_atari.base_stochastic_env_classes import ActionDependentStochasticityWrapper, ActionIndependentRandomStochasticityWrapper, ActionIndependentConceptDriftWrapper, PartialObservationWrapper
from stochastic_atari.utils import update_ram_state, blackout_obs, _save_restore_handler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######################### RAM utils for BankHeist #########################
BankHeist_RAM_Mapping = {
    # general states
    'lives': 85,
    'bomb': 12, # maybe 0 to defuse (doubtful)

    # robber car states
    'robber_car_x_pos': 28, # 12-140
    'robber_car_y_pos': 8,
    'robber_car_hide_states': {8: 155, 28: 12},
    'robber_car_fuel': 86,
    'robber_car_fuel_range': (0, 25), # 0 = full, 25 = empty
    'robber_car_direction': 15,
    'robber_car_direction_states': {
        'forward': {15: 0},
        'reverse': {15: 254},
    },
    'robber_car_orientation': 51,
    'robber_car_orientation_states': {
        'horizontal': {51: 0},
        'vertical': {51: 12},
    },

    # city states
    'switch_between_cities': 0, # 0-3
    'bg_color': 69,
    'city_bg_color_blend_states': {
        '0': {69: 30},
        '1': {69: 190},
        '2': {69: 46},
        '3': {69: 126},
    },
    'city_blocks_disappear': {66:0, 68:0},

    # bank/police states
    'bank_police_status': [24, 25, 26],
    'bank_police_status_states': {
        'bank': 253,
        'police': 254,
        'hide': 255,
    },
    'police_car_orientation': [52, 53, 54],
    'police_car_orientation_states': {
        'horizontal': {52: 0, 53: 0, 54: 0},
        'vertical': {52: 12, 53: 12, 54: 12},
    },
}

######################### Partial observation utils for BankHeist #########################

TOP_WALL_REGION = {'min_x': np.int64(12),
                    'max_x': np.int64(147),
                    'min_y': np.int64(37),
                    'max_y': np.int64(44)}

LEFT_WALL_REGION = {'min_x': np.int64(12),
                    'max_x': np.int64(15),
                    'min_y': np.int64(37),
                    'max_y': np.int64(177)}

BOTTOM_WALL_REGION = {'min_x': np.int64(12),
                        'max_x': np.int64(147),
                        'min_y': np.int64(173),
                        'max_y': np.int64(177)}

RIGHT_WALL_REGION = {'min_x': np.int64(144),
                    'max_x': np.int64(147),
                    'min_y': np.int64(37),
                    'max_y': np.int64(177)}

CITY_WALL_REGIONS = {'TOP': TOP_WALL_REGION,
                     'LEFT': LEFT_WALL_REGION,
                     'BOTTOM': BOTTOM_WALL_REGION,
                     'RIGHT': RIGHT_WALL_REGION}

FUEL_REGION = {'min_x': np.int64(12),
                'max_x': np.int64(73),
                'min_y': np.int64(11),
                'max_y': np.int64(36)}

LIVES_REGION = {'min_x': np.int64(78),
                'max_x': np.int64(147),
                'min_y': np.int64(11),
                'max_y': np.int64(36)}

SCORE_REGION = {'min_x': np.int64(12),
                'max_x': np.int64(147),
                'min_y': np.int64(178),
                'max_y': np.int64(186)}

FUEL_LIVES_REGION_COLOR = [0, 0, 148]
SCORE_REGION_COLOR = [162, 98, 33]

blackout_regions = [CITY_WALL_REGIONS, FUEL_REGION, LIVES_REGION, SCORE_REGION]

def blackout_city_wall_regions(array, wall_regions, top=True, left=True, bottom=True, right=True):
    """
    Black out the city wall regions of the observation.
    """
    if top:
        array = blackout_obs(array, wall_regions['TOP'])
    if left:
        array = blackout_obs(array, wall_regions['LEFT'])
    if bottom:
        array = blackout_obs(array, wall_regions['BOTTOM'])
    if right:
        array = blackout_obs(array, wall_regions['RIGHT'])
    return array


def blackout_obs_mode(array, mode='0'):
    """
    Black out regions of the observation according to the selected mode for Bank Heist.

    Modes:
        '0': none - Do not black out any region.
        '1': all - Black out all defined regions (city walls, fuel, lives, score).
        '2': city walls - Black out all city wall regions (top, left, bottom, right).
        '3': top city wall - Black out only the top city wall region.
        '4': left city wall - Black out only the left city wall region.
        '5': bottom city wall - Black out only the bottom city wall region.
        '6': right city wall - Black out only the right city wall region.
        '7': left and right city walls together - Black out only the left and right city walls together.
        '8': fuel - Black out the fuel region.
        '9': lives - Black out the lives region.
        '10': score - Black out the score region.

    Args:
        array: numpy array of shape (height, width, 3)
        mode: str, one of ['0', '1', ..., '10']

    Returns:
        array: numpy array with selected regions blacked out
    """
    if mode == '0':  # none
        return array
    elif mode == '1':  # all
        # Black out all regions: city walls, fuel, lives, score
        for region in blackout_regions:
            if region == CITY_WALL_REGIONS:
                array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=True, left=True, bottom=True, right=True)
            elif region == FUEL_REGION:
                array = blackout_obs(array, FUEL_REGION, color=FUEL_LIVES_REGION_COLOR)
            elif region == LIVES_REGION:
                array = blackout_obs(array, LIVES_REGION, color=FUEL_LIVES_REGION_COLOR)
            elif region == SCORE_REGION:
                array = blackout_obs(array, SCORE_REGION, color=SCORE_REGION_COLOR)
        return array
    elif mode == '2':  # city walls (all)
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=True, left=True, bottom=True, right=True)
        return array
    elif mode == '3':  # top city wall
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=True, left=False, bottom=False, right=False)
        return array
    elif mode == '4':  # left city wall
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=False, left=True, bottom=False, right=False)
        return array
    elif mode == '5':  # bottom city wall
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=False, left=False, bottom=True, right=False)
        return array
    elif mode == '6':  # right city wall
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=False, left=False, bottom=False, right=True)
        return array
    elif mode == '7':  # left and right city walls together
        array = blackout_city_wall_regions(array, CITY_WALL_REGIONS, top=False, left=True, bottom=False, right=True)
        return array
    elif mode == '8':  # fuel
        array = blackout_obs(array, FUEL_REGION, color=FUEL_LIVES_REGION_COLOR)
        return array
    elif mode == '9':  # lives
        array = blackout_obs(array, LIVES_REGION, color=FUEL_LIVES_REGION_COLOR)
        return array
    elif mode == '10':  # score
        array = blackout_obs(array, SCORE_REGION, color=SCORE_REGION_COLOR)
        return array
    else:
        raise ValueError(f"Unknown blackout mode: {mode}")

######################### Ram modification utils for BankHeist #########################

def ram_obs_modification_mode(env, mode='0', get_obs=True, verbose=False):
    """
    Modify the RAM of the BankHeist environment according to the selected mode.

    Modes:
        '0': none - Do not modify RAM.
        '1': hide robber's car - Hide the robber's car in RAM.
        '2': hide change in fuel (show always full) - Keep the fuel indicator always full in RAM.
        '3': hide city blocks - Hide the city blocks in RAM.
        '4': hide city blocks and wall (blend) - Hide both city blocks and city wall, blending them in RAM.
        '5': hide banks - Hide the banks in RAM.
        '6': hide police - Hide the police in RAM.

    Args:
        env: BankHeist environment.
        mode: str, one of ['0', '1', '2', '3', '4', '5', '6'].
        get_obs: bool, whether to return the observation after modification.
        verbose: bool, whether to print debug info.

    Returns:
        None or modified observation, depending on get_obs.
    """

    if mode == '0':
        return 

    elif mode == '1':  # hide robber's car
        update_ram_dict = BankHeist_RAM_Mapping['robber_car_hide_states']
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '2':  # hide change in fuel (show always full)
        update_ram_dict = {BankHeist_RAM_Mapping['robber_car_fuel']: 0}
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '3':  # hide city blocks
        update_ram_dict = BankHeist_RAM_Mapping['city_blocks_disappear']
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '4':  # hide city blocks and wall (blend)
        ram = env.unwrapped.ale.getRAM()
        city_id = int(ram[BankHeist_RAM_Mapping['switch_between_cities']]) % 4
        city_id_str = str(city_id)
        update_ram_dict = BankHeist_RAM_Mapping['city_bg_color_blend_states'][city_id_str]
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '5':  # hide banks (only if current state is bank)
        ram = env.unwrapped.ale.getRAM()
        update_ram_dict = {}
        for addr in BankHeist_RAM_Mapping['bank_police_status']:
            if int(ram[addr]) == BankHeist_RAM_Mapping['bank_police_status_states']['bank']:
                update_ram_dict[addr] = BankHeist_RAM_Mapping['bank_police_status_states']['hide']
        if not update_ram_dict:
            return 
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    elif mode == '6':  # hide police (only if current state is police)
        ram = env.unwrapped.ale.getRAM()
        update_ram_dict = {}
        for addr in BankHeist_RAM_Mapping['bank_police_status']:
            if int(ram[addr]) == BankHeist_RAM_Mapping['bank_police_status_states']['police']:
                update_ram_dict[addr] = BankHeist_RAM_Mapping['bank_police_status_states']['hide']
        if not update_ram_dict:
            return 
        return _save_restore_handler(env, update_ram_dict, get_obs=get_obs, verbose=verbose)

    else:
        raise ValueError(f"Unknown RAM modification mode: {mode}")

######################### Action independent random stochasticity wrapper for BankHeist #########################

class BankHeistActionIndependentRandomStochasticityWrapper(ActionIndependentRandomStochasticityWrapper):
    """
    Wrapper that implements action independent random stochasticity in internal ale _env for BankHeist.

    Modes:
        mode '0': No stochasticity.
        mode '1': Dropped bomb is a dud (bomb does not explode or have effect).
        mode '2': Fuel leaks (fuel suddenly drops to a low value, once per city per episode).
        mode '3': Switches city mid way (player is teleported to a different city unexpectedly).
        mode '4': Bank was empty (no reward is given when looting the bank, only if change from bank to police is detected).
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self._original_ram_state = self.env.unwrapped.original_ram_state
        self._previous_ram_state = self._original_ram_state.copy()
        self._fuel_leak_cities = set()

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self._original_ram_state = self.env.unwrapped.ale.getRAM().copy()
        self._previous_ram_state = self._original_ram_state.copy()
        self._fuel_leak_cities = set()
        return obs

    def _bomb_dud(self, action, verbose=False):
        """
        Dropped bomb is a dud (bomb does not explode or have effect).
        Only check probability when a change in bomb addr is noticed.
        """
        if self._previous_ram_state is None:
            self._previous_ram_state = self.env.unwrapped.ale.getRAM().copy()
        prev_ram = self._previous_ram_state
        obs, reward, done, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        bomb_addr = BankHeist_RAM_Mapping['bomb']
        if ram[bomb_addr] != self._original_ram_state[bomb_addr]:
            if prev_ram[bomb_addr] != ram[bomb_addr]:
                if np.random.random() < self.prob:
                    update_ram_state(self.env, {bomb_addr: prev_ram[bomb_addr]})
                    if verbose:
                        print(f"Bomb dud: bomb RAM reverted from {ram[bomb_addr]} to {prev_ram[bomb_addr]}")
        self._previous_ram_state = ram.copy()
        return obs, reward, done, truncated, info

    def _fuel_leak(self, action, verbose=False):
        """
        Fuel leaks (fuel suddenly drops to a low value).
        Only do this once per city per episode. (idel prob is 0.001)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        city_addr = BankHeist_RAM_Mapping['switch_between_cities']
        city_id = int(ram[city_addr]) % 4
        if city_id not in self._fuel_leak_cities and np.random.random() < self.prob:
            fuel_addr = BankHeist_RAM_Mapping['robber_car_fuel']
            _, fuel_max = BankHeist_RAM_Mapping['robber_car_fuel_range']
            leak_value = fuel_max//2
            if leak_value > ram[fuel_addr]:
                update_ram_state(self.env, {fuel_addr: leak_value})
            if verbose:
                print(f"Fuel leak: city {city_id}, fuel set to {leak_value}")
            self._fuel_leak_cities.add(city_id)
        return obs, reward, done, truncated, info

    def _city_switch(self, action, verbose=False):
        """
        Switches city mid way (player is teleported to a different city unexpectedly).
        No restriction on number of times per episode. (idel prob is 0.001)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        if np.random.random() < self.prob:
            city_addr = BankHeist_RAM_Mapping['switch_between_cities']
            ram = self.env.unwrapped.ale.getRAM()
            current_city = int(ram[city_addr]) % 4
            possible_cities = [i for i in range(4) if i != current_city]
            new_city = np.random.choice(possible_cities)
            update_ram_state(self.env, {city_addr: new_city})
            if verbose:
                print(f"City switch: city changed from {current_city} to {new_city}")
        return obs, reward, done, truncated, info

    def _bank_empty(self, action, verbose=False):
        """
        Bank was empty (no reward is given when looting the bank).
        Only return empty if change from bank to police is detected compared to prev ram state.
        """
        if self._previous_ram_state is None:
            self._previous_ram_state = self.env.unwrapped.ale.getRAM().copy()
        prev_ram = self._previous_ram_state
        obs, reward, done, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        bank_status_addrs = BankHeist_RAM_Mapping['bank_police_status']
        bank_state = BankHeist_RAM_Mapping['bank_police_status_states']['bank']
        change_detected = False
        for addr in bank_status_addrs:
            if int(prev_ram[addr]) == bank_state and int(ram[addr]) != bank_state:
                change_detected = True
                break
        if change_detected and np.random.random() < self.prob:
            if verbose:
                print(f"Bank empty: reward set to 0 (was {reward})")
            reward = 0.0
        self._previous_ram_state = ram.copy()
        return obs, reward, done, truncated, info

    def step(self, action):
        """
        Implements the action-independent random stochasticity for BankHeist.

        Modes:
            mode '0': No stochasticity.
            mode '1': Dropped bomb is a dud (bomb does not explode or have effect).
            mode '2': Fuel leaks (fuel suddenly drops to a low value, once per city per episode).
            mode '3': Switches city mid way (player is teleported to a different city unexpectedly).
            mode '4': Bank was empty (no reward is given when looting the bank, only if change from bank to police is detected).
        """
        if self.mode == '0':
            return self.env.step(action)
        elif self.mode == '1':
            return self._bomb_dud(action, verbose=True)
        elif self.mode == '2':
            return self._fuel_leak(action, verbose=True)
        elif self.mode == '3':
            return self._city_switch(action, verbose=True)
        elif self.mode == '4':
            return self._bank_empty(action, verbose=True)
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")


######################### Partial observation wrapper for BankHeist #########################

class BankHeistPartialObservationWrapper(PartialObservationWrapper):
    def _blackout_obs_mode(self, array, mode) -> None:
        blackout_obs_mode(array, mode)
    
    def _ram_obs_mode(self, env, mode) -> np.ndarray:
        return ram_obs_modification_mode(env, mode, get_obs=True, verbose=False)

######################### Action dependent stochasticity wrapper for BankHeist #########################

class BankHeistActionDependentStochasticityWrapper(ActionDependentStochasticityWrapper):
    """
    Wrapper that implements action dependent stochasticity in internal ale _env for BankHeist.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.prob = config['stochastic_action_prob']
        # this is to decrease the number of fire based actions
        # the agent can take during random action selection
        # to avoid agent instantly dying due to fire action
        self.restrict_fire_actions = True
        if self.restrict_fire_actions:
            print("restricting fire actions during random action selection")

    def action(self, action):
        logger.debug(f"using action() method of ActionDependentStochasticityWrapper")
        if np.random.random() < self.prob:
            # Choose random action
            if self.restrict_fire_actions:
                # restrict to first 10 actions
                # Restrict sampling to first 10 actions (i.e., actions 0-9)
                from gymnasium.spaces import Discrete
                sample_action_space = Discrete(10)
            else:
                sample_action_space = self.env.action_space
            return sample_action_space.sample()
        else:
            # Use predicted action
            return action
######################### Wrapper registry for BankHeist #########################

def get_bankheist_wrapper_registry():
    return {
        'action_dependent': BankHeistActionDependentStochasticityWrapper,
        'action_independent_random': BankHeistActionIndependentRandomStochasticityWrapper,
        'action_independent_concept_drift': ActionIndependentConceptDriftWrapper,
        'partial_observation': BankHeistPartialObservationWrapper,
    }
