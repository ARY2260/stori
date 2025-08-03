import numpy as np

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

def blackout_obs(array, region):
    array[region['min_y']:region['max_y']+1, region['min_x']:region['max_x']+1] = [0, 0, 0]
    return array

def crop_left_right(array, type='left'):
    height, width = array.shape[:2]
    if type == 'left':
        array[:, :width//2, :] = 0
    elif type == 'right':
        array[:, width//2:, :] = 0
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

def random_circular_mask_blocks_obs(array, radius=10):
    assert Blocks_region['min_x']+radius < Blocks_region['max_x']-radius
    assert Blocks_region['min_y']+radius < Blocks_region['max_y']-radius
    center_x = np.random.randint(Blocks_region['min_x']+radius, Blocks_region['max_x']-radius)
    center_y = np.random.randint(Blocks_region['min_y']+radius, Blocks_region['max_y']-radius)
    array = _circular_mask_obs(array, center_x=center_x, center_y=center_y, radius=radius, invert=True)
    return array

def random_circular_mask_obs(array, radius=75):
    # randomly choose center_x and center_y from obs shape
    center_x = np.random.randint(0, array.shape[1])
    center_y = np.random.randint(0, array.shape[0])
    array = _circular_mask_obs(array, center_x=center_x, center_y=center_y, radius=radius, invert=False)
    return array

