from envs.post_processing_utils import *

# give different modes with region choices for blackout
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

def crop_obs_mode(array, mode='0'):
    """
    Crop the observation according to the selected mode.

    Modes:
        '0': none - Do not crop any region
        '1': left - Crop the left half of the observation
        '2': right - Crop the right half of the observation
        '3': random circular mask blocks - Randomly mask a circular region of the blocks
        '4': random circular mask - Randomly mask a circular region of the observation
    """
    if mode == '0':  # none
        pass  # do nothing
    elif mode == '1':  # left
        array = crop_left_right(array, type='left')
    elif mode == '2':  # right
        array = crop_left_right(array, type='right')
    # circular mask
    elif mode == '3':  # random circular mask blocks
        array = random_circular_mask_blocks_obs(array)
    elif mode == '4':  # random circular mask
        array = random_circular_mask_obs(array)
    else:
        raise ValueError(f"Unknown crop mode: {mode}")
    return array
