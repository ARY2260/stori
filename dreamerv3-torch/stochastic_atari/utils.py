import numpy as np
import logging

logger = logging.getLogger(__name__)


def crop_left_right_top_bottom(array, type='left'):
    height, width = array.shape[:2]
    if type == 'left':
        array[:, :width//2, :] = 0
    elif type == 'right':
        array[:, width//2:, :] = 0
    elif type == 'top':
        array[:height//2, :, :] = 0
    elif type == 'bottom':
        array[height//2:, :, :] = 0
    else:
        raise ValueError(f"Unknown crop type: {type}")
    return array

def _circular_mask_obs(array, center_x, center_y, radius, invert=False):
    """
    Create a circular mask on the observation and black out pixels outside the circle.
    
    Args:
        obs: numpy array of shape (height, width, 3) - the observation image
        center_x: x-coordinate of the circle center
        center_y: y-coordinate of the circle center  
        radius: radius of the circle in pixels
        invert: if True, black out pixels inside the circle
    
    Returns:
        modified_obs: observation with pixels outside the circle set to black [0, 0, 0]
    """
    
    # Create a copy of the observation
    height, width = array.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Calculate distance from center for each pixel
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create mask for pixels outside the circle
    outside_circle_mask = distances > radius
    
    # Set pixels outside the circle to black
    if invert:
        array[~outside_circle_mask] = [0, 0, 0]
    else:
        array[outside_circle_mask] = [0, 0, 0]
    
    return array

def random_circular_mask_obs(array, radius=75):
    # randomly choose center_x and center_y from obs shape
    center_x = np.random.randint(0, array.shape[1])
    center_y = np.random.randint(0, array.shape[0])
    array = _circular_mask_obs(array, center_x=center_x, center_y=center_y, radius=radius, invert=False)
    return array

def crop_obs_mode(array, mode='0'):
    """
    Crop the observation according to the selected mode.

    Modes:
        '0': none - Do not crop any region
        '1': left - Crop the left half of the observation
        '2': right - Crop the right half of the observation
        '3': top - Crop the top half of the observation 
        '4': bottom - Crop the bottom half of the observation
        '5': random circular mask - Randomly mask a circular region of the observation
    """
    if mode == '0':  # none
        pass  # do nothing
    elif mode == '1':  # left
        array = crop_left_right_top_bottom(array, type='left')
    elif mode == '2':  # right
        array = crop_left_right_top_bottom(array, type='right')
    elif mode == '3':  # top
        array = crop_left_right_top_bottom(array, type='top')
    elif mode == '4':  # bottom
        array = crop_left_right_top_bottom(array, type='bottom')
    elif mode == '5':  # random circular mask
        array = random_circular_mask_obs(array)
    else:
        raise ValueError(f"Unknown crop mode: {mode}")
    return array


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
