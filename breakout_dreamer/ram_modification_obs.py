from ram_modification_utils import *


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

    elif mode == '1':  # nus_pattern
        # save env state before modification
        saved_state = env.unwrapped.clone_state(True)
        update_ram_state(env, NUS_pattern_blocks_ram_mapping, verbose=verbose)
        obs, _, _, _ = env.step(0)
        # restore env state
        env.unwrapped.restore_state(saved_state)
        # env.step(0) #optional
        return obs

    elif mode == '2':  # ball_hidden
        # save env state before modification
        saved_state = env.unwrapped.clone_state(True)
        update_ram_state(env, Ball_hidden_ram_mapping, verbose=verbose)
        obs, _, _, _ = env.step(0)
        # restore env state
        env.unwrapped.restore_state(saved_state)
        # env.step(0) #optional
        return obs

    else:
        raise ValueError(f"Unknown RAM modification mode: {mode}")
