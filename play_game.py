import gymnasium
from gymnasium.utils.play import play
import ale_py
import pygame
from stori import create_stochasticity_profile

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

    "bankheist": {
            # (): 0,  # NOOP
            (pygame.K_SPACE,): 1,  # FIRE
            (pygame.K_UP,): 2,  # UP
            (pygame.K_RIGHT,): 3,  # RIGHT
            (pygame.K_LEFT,): 4,   # LEFT
            (pygame.K_DOWN,): 5,  # DOWN
            (pygame.K_UP, pygame.K_RIGHT): 6,   # UPRIGHT
            (pygame.K_UP, pygame.K_LEFT): 7,    # UPLEFT
            (pygame.K_DOWN, pygame.K_RIGHT): 8, # DOWNRIGHT
            (pygame.K_DOWN, pygame.K_LEFT): 9,  # DOWNLEFT
            (pygame.K_UP, pygame.K_SPACE): 10,    # UPFIRE
            (pygame.K_RIGHT, pygame.K_SPACE): 11, # RIGHTFIRE
            (pygame.K_LEFT, pygame.K_SPACE): 12,  # LEFTFIRE
            (pygame.K_DOWN, pygame.K_SPACE): 13,  # DOWNFIRE
            (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,   # UPRIGHTFIRE
            (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,    # UPLEFTFIRE
            (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16, # DOWNRIGHTFIRE
            (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,  # DOWNLEFTFIRE
            },
    }

stochasticity_config = {'stochasticity_type': '3.2',
                        'intrinsic_stochasticity': {
                            'action_dependent': {
                                'stochastic_action_prob': 0.5
                                },
                            'action_independent_concept_drift': {
                                'temporal_threshold': 300,
                                'temporal_mode': 'cyclic',
                                'secondary_concept_type': "3.2",
                                },
                            'action_independent_random': {
                                'mode': '3',
                                'random_stochasticity_prob': 0.25,
                                }
                            },
                        'partial_observation': {
                            'type': 'ram',
                            'mode': '4', 
                            'prob': 0.75, 
                            },
                        }


# env_name = "ALE/Boxing-v5"
# gymnasium.register_envs(ale_py)
# env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
# game_name = env_name[4:-3].lower() # Remove first 4 characters ("ALE/") and last 3 characters ("-v5")
# stochasticity_profile = create_stochasticity_profile(game_name=game_name, type=stochasticity_config['stochasticity_type'], config=stochasticity_config)
# env = stochasticity_profile.get_env(env)

# play(env, keys_to_action=keys_to_action[game_name])

import tkinter as tk
from tkinter import ttk

import json
import os

def launch_mode_selector():
    # Load experiment configs from Exp_configs.json
    exp_config_path = os.path.join(os.path.dirname(__file__), "Exp_configs.json")
    with open(exp_config_path, "r") as f:
        exp_configs = json.load(f)

    # Only allow these 4 games (as in the config)
    allowed_games = [
        ("Breakout", "breakout"),
        ("Boxing", "boxing"),
        ("Gopher", "gopher"),
        ("BankHeist", "bankheist"),
    ]
    allowed_game_names = [g[0] for g in allowed_games]

    # Map game name to ALE env name
    def game_to_env_name(game):
        return f"ALE/{game}-v5"

    # Helper: flatten nested config dict for editing
    def flatten_config(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Helper: unflatten config dict
    def nested_dict_from_flat(flat):
        nested = {}
        for k, v in flat.items():
            parts = k.split('.')
            d = nested
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = v
        return nested

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Select Experiment Condition")

    frm = tk.Frame(root)
    frm.pack(padx=10, pady=10)

    # Game selection dropdown
    tk.Label(frm, text="Game").grid(row=0, column=0, sticky="e", padx=5, pady=2)
    game_var = tk.StringVar(value=allowed_game_names[0])
    game_menu = ttk.Combobox(frm, values=allowed_game_names, state="readonly", textvariable=game_var)
    game_menu.current(0)
    game_menu.grid(row=0, column=1, padx=5, pady=2)

    # Type selection dropdown (will be updated based on game)
    tk.Label(frm, text="Type").grid(row=1, column=0, sticky="e", padx=5, pady=2)
    type_var = tk.StringVar()
    type_menu = ttk.Combobox(frm, state="readonly", textvariable=type_var)
    type_menu.grid(row=1, column=1, padx=5, pady=2)

    # Config fields (will be updated based on type)
    config_entries = {}

    # --- Use this for type reference ---
    # Reference config for type inference
    reference_stochasticity_config = {
        'stochasticity_type': '3.2',
        'intrinsic_stochasticity': {
            'action_dependent': {
                'stochastic_action_prob': 0.5
            },
            'action_independent_concept_drift': {
                'temporal_threshold': 300,
                'temporal_mode': 'cyclic',
                'secondary_concept_type': "3.2",
            },
            'action_independent_random': {
                'mode': '3',
                'random_stochasticity_prob': 0.25,
            }
        },
        'partial_observation': {
            'type': 'ram',
            'mode': '4',
            'prob': 0.75,
        },
    }
    reference_flat = flatten_config(reference_stochasticity_config)

    # Update type options when game changes
    def update_types(*args):
        game_display = game_var.get()
        type_options = list(exp_configs[game_display].keys())
        type_menu['values'] = type_options
        type_menu.current(0)
        update_config_fields()

    # Update config fields when type changes
    def update_config_fields(*args):
        # Clear old entries
        for widget in config_entries.values():
            widget[0].destroy()
            widget[1].destroy()
        config_entries.clear()

        game_display = game_var.get()
        type_name = type_var.get()
        if not type_name:
            type_name = list(exp_configs[game_display].keys())[0]
        config_flat = flatten_config(exp_configs[game_display][type_name])

        # Show all config fields for this type
        for i, (key, val) in enumerate(config_flat.items()):
            lbl = tk.Label(frm, text=key)
            lbl.grid(row=2+i, column=0, sticky="e", padx=5, pady=2)
            ent = tk.Entry(frm)
            ent.insert(0, str(val))
            ent.grid(row=2+i, column=1, padx=5, pady=2)
            config_entries[key] = (lbl, ent)

    # Bind game and type selection
    game_menu.bind("<<ComboboxSelected>>", lambda e: update_types())
    type_menu.bind("<<ComboboxSelected>>", lambda e: update_config_fields())

    # Initialize type options and config fields
    update_types()

    def smart_cast(val, ref_val):
        """
        Cast val to the type of ref_val, if possible.
        """
        if isinstance(ref_val, bool):
            # Accept "True"/"False" (case-insensitive) or 1/0
            if isinstance(val, str):
                if val.lower() == "true":
                    return True
                if val.lower() == "false":
                    return False
            try:
                return bool(int(val))
            except Exception:
                return bool(val)
        elif isinstance(ref_val, int) and not isinstance(ref_val, bool):
            try:
                return int(val)
            except Exception:
                return val
        elif isinstance(ref_val, float):
            try:
                return float(val)
            except Exception:
                return val
        else:
            return val

    def on_play():
        # Gather values from entries
        user_config = {}
        # Get the reference config for type inference
        game_display = game_var.get()
        type_name = type_var.get()
        if not type_name:
            type_name = list(exp_configs[game_display].keys())[0]
        config_flat = flatten_config(exp_configs[game_display][type_name])

        for key, (lbl, ent) in config_entries.items():
            val = ent.get()
            # Use type from reference_flat if available, else from config_flat, else fallback
            if key in reference_flat:
                ref_val = reference_flat[key]
            elif key in config_flat:
                ref_val = config_flat[key]
            else:
                ref_val = None

            if ref_val is not None:
                user_config[key] = smart_cast(val, ref_val)
            else:
                # Fallback: try to cast to float/int, else string
                try:
                    if '.' in val:
                        user_config[key] = float(val)
                    else:
                        user_config[key] = int(val)
                except Exception:
                    user_config[key] = val

        # Convert flat dict to nested config
        stochasticity_config = nested_dict_from_flat(user_config)

        # Get selected game and type
        game_name = [g[1] for g in allowed_games if g[0] == game_display][0]
        env_name = game_to_env_name(game_display)

        # Set the type for the profile (from config)
        stoch_type = stochasticity_config.get('stochasticity_type', None)
        if stoch_type is None:
            # fallback: try from exp_configs
            stoch_type = exp_configs[game_display][type_name].get('stochasticity_type', None)
        # If still None, fallback to type_name (e.g. "3.2")
        if stoch_type is None:
            stoch_type = type_name.split()[-1] if " " in type_name else type_name

        global env, stochasticity_profile
        stoch_type = str(stoch_type)
        stochasticity_profile = create_stochasticity_profile(
            game_name=game_name,
            type=stoch_type,
            config=stochasticity_config
        )

        gymnasium.register_envs(ale_py)
        env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
        env = stochasticity_profile.get_env(env)
        root.destroy()
        # Wrap play to re-show panel after game closes
        def play_and_reopen(*args, **kwargs):
            import pygame
            pygame.init()
            try:
                pygame.display.set_mode((1280,960))
            except Exception:
                pass
            play(env, keys_to_action=keys_to_action[game_name])
            launch_mode_selector()
        play_and_reopen()

    play_button = tk.Button(root, text="Play", command=on_play)
    play_button.pack(pady=10)

    root.mainloop()

launch_mode_selector()






