import gymnasium
from gymnasium.utils.play import play
import ale_py
import pygame
from stochastic_atari import create_stochasticity_profile

keys_to_action = {
     "gopher": {
                    # (): 0,  # NOOP
                    (pygame.K_SPACE,): 1,  # FIRE
                    (pygame.K_UP,): 2,  # UP
                    (pygame.K_RIGHT,): 3,  # RIGHT
                    (pygame.K_LEFT,): 4,   # LEFT
                    (pygame.K_UP, pygame.K_SPACE): 5,     # UPFIRE
                    (pygame.K_RIGHT, pygame.K_SPACE): 6,  # RIGHTFIRE
                    (pygame.K_LEFT, pygame.K_SPACE): 7,   # LEFTFIRE
                },
    "boxing":  {
                # (): 0,                                         # NOOP
                (pygame.K_SPACE,): 1,                          # FIRE  (straight punch)
                (pygame.K_UP,): 2,                             # UP
                (pygame.K_RIGHT,): 3,                          # RIGHT
                (pygame.K_LEFT,): 4,                           # LEFT
                (pygame.K_DOWN,): 5,                           # DOWN
                # diagonals (no punch)
                (pygame.K_UP,   pygame.K_RIGHT): 6,            # UPRIGHT
                (pygame.K_UP,   pygame.K_LEFT): 7,             # UPLEFT
                (pygame.K_DOWN, pygame.K_RIGHT): 8,            # DOWNRIGHT
                (pygame.K_DOWN, pygame.K_LEFT): 9,             # DOWNLEFT
                # single direction + punch
                (pygame.K_UP,    pygame.K_SPACE): 10,          # UPFIRE
                (pygame.K_RIGHT, pygame.K_SPACE): 11,          # RIGHTFIRE
                (pygame.K_LEFT,  pygame.K_SPACE): 12,          # LEFTFIRE
                (pygame.K_DOWN,  pygame.K_SPACE): 13,          # DOWNFIRE
                # diagonal punches
                (pygame.K_UP,   pygame.K_RIGHT, pygame.K_SPACE): 14,  # UPRIGHTFIRE
                (pygame.K_UP,   pygame.K_LEFT,  pygame.K_SPACE): 15,  # UPLEFTFIRE
                (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,  # DOWNRIGHTFIRE
                (pygame.K_DOWN, pygame.K_LEFT,  pygame.K_SPACE): 17,  # DOWNLEFTFIRE
            },
    "breakout": {
                # (): 0,
                (pygame.K_SPACE,): 1,
                (pygame.K_RIGHT,): 2,
                (pygame.K_LEFT,): 3,
            },
    }

stochasticity_config = {'stochasticity_type': 4,
                        'intrinsic_stochasticity': {
                            'action_dependent': {
                                'stochastic_action_prob': 0.5
                                },
                            'action_independent_concept_drift': {
                                'temporal_threshold': 300,
                                'temporal_mode': 'cyclic',
                                'secondary_concept_type': 5,
                                },
                            'action_independent_random': {
                                'mode': '2',
                                'random_stochasticity_prob': 0.25,
                                }
                            },
                        'partial_observation': {
                            'type': 'ram',
                            'mode': '4', 
                            'prob': 0.75, 
                            },
                        }


env_name = "ALE/Gopher-v5" # "ALE/Boxing-v5" # "ALE/Breakout-v5"
gymnasium.register_envs(ale_py)
env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
game_name = env_name[4:-3].lower() # Remove first 4 characters ("ALE/") and last 3 characters ("-v5")
stochasticity_profile = create_stochasticity_profile(game_name=game_name, type=stochasticity_config['stochasticity_type'], config=stochasticity_config)
env = stochasticity_profile.get_env(env)

play(env, keys_to_action=keys_to_action[game_name])