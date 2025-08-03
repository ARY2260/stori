import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def _update_ram_state_index(env, _index, newVal, verbose=True):
    if verbose:
        ram = env.unwrapped.ale.getRAM()
        logger.info("before updating ram index %d, val %d", _index, ram[_index])

    env.unwrapped.ale.setRAM(_index, newVal)

    if verbose:
        ramN = env.unwrapped.ale.getRAM()
        logger.info("after updating ram index %d, val %d", _index, ramN[_index])

def update_ram_state(env, update_ram_dict, verbose=False):
    for _index, newVal in update_ram_dict.items():
        _update_ram_state_index(env, _index, newVal, verbose)


def calculate_power_of_2_scale_factor(max_binary_value, target_max=255):
    """
    Calculate scale factor as a power of 2 that fits within target_max
    
    Returns the largest power of 2 that when multiplied by max_binary_value
    doesn't exceed target_max
    """
    import math
    
    # Find the largest power of 2 that fits
    max_scale = target_max // max_binary_value
    power_of_2_scale = 2 ** int(math.log2(max_scale))
    
    return power_of_2_scale

def binary_to_scaled_decimal(binary_string, reverse=True, target_max=255):
    num_bits = len(binary_string)
    max_binary_value = (2 ** num_bits) - 1
    
    # Calculate scale factor as power of 2
    scale_factor = calculate_power_of_2_scale_factor(max_binary_value, target_max)
    
    if reverse:
        binary_string = binary_string[::-1]
    
    decimal = int(binary_string, 2)
    scaled = decimal * scale_factor
    return scaled

