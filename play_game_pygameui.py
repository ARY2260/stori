import asyncio
import pygame
import sys
import traceback
import os
import json
from datetime import datetime

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 215, 0)
ERROR_COLOR = (255, 50, 50)

# --- Global Placeholders ---
# We define these globally so we can load them async inside main
gymnasium = None
ale_py = None
create_stochasticity_profile = None

keys_to_action = {
     "gopher": {
                    (pygame.K_SPACE,): 1, (pygame.K_UP,): 2, (pygame.K_RIGHT,): 3, (pygame.K_LEFT,): 4,
                    (pygame.K_UP, pygame.K_SPACE): 5, (pygame.K_RIGHT, pygame.K_SPACE): 6, (pygame.K_LEFT, pygame.K_SPACE): 7,
                },
    "boxing":  {
                (pygame.K_SPACE,): 1, (pygame.K_UP,): 2, (pygame.K_RIGHT,): 3, (pygame.K_LEFT,): 4, (pygame.K_DOWN,): 5,
                (pygame.K_UP, pygame.K_RIGHT): 6, (pygame.K_UP, pygame.K_LEFT): 7, (pygame.K_DOWN, pygame.K_RIGHT): 8,
                (pygame.K_DOWN, pygame.K_LEFT): 9, (pygame.K_UP, pygame.K_SPACE): 10, (pygame.K_RIGHT, pygame.K_SPACE): 11,
                (pygame.K_LEFT,  pygame.K_SPACE): 12, (pygame.K_DOWN,  pygame.K_SPACE): 13, (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
                (pygame.K_UP, pygame.K_LEFT,  pygame.K_SPACE): 15, (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16, (pygame.K_DOWN, pygame.K_LEFT,  pygame.K_SPACE): 17,
            },
    "breakout": { (pygame.K_SPACE,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3 },
    "bankheist": {
            (pygame.K_SPACE,): 1, (pygame.K_UP,): 2, (pygame.K_RIGHT,): 3, (pygame.K_LEFT,): 4, (pygame.K_DOWN,): 5,
            (pygame.K_UP, pygame.K_RIGHT): 6, (pygame.K_UP, pygame.K_LEFT): 7, (pygame.K_DOWN, pygame.K_RIGHT): 8, (pygame.K_DOWN, pygame.K_LEFT): 9,
            (pygame.K_UP, pygame.K_SPACE): 10, (pygame.K_RIGHT, pygame.K_SPACE): 11, (pygame.K_LEFT, pygame.K_SPACE): 12, (pygame.K_DOWN, pygame.K_SPACE): 13,
            (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14, (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15, (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,
            (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,
            },
}

# --- Helper Functions ---

def nested_dict_from_flat(flat):
    """Convert flat dict with dot-separated keys to nested dict."""
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

def draw_text_centered(screen, text, y_offset=0, size=40, color=TEXT_COLOR):
    font = pygame.font.Font(None, size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset))
    screen.blit(surf, rect)

def draw_error_screen(screen, error_msg):
    screen.fill((0, 0, 0))
    font_title = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 24)
    
    title = font_title.render("Startup Error", True, ERROR_COLOR)
    screen.blit(title, (20, 20))
    
    y = 70
    for line in error_msg.split('\n'):
        # Wrap simple text
        if len(line) > 80:
            line = line[:80] + "..."
        text = font_small.render(line, True, TEXT_COLOR)
        screen.blit(text, (20, y))
        y += 25
        if y > SCREEN_HEIGHT - 20: break
    pygame.display.flip()

class Menu:
    def __init__(self, screen, options, title="Select Option"):
        self.screen = screen
        self.options = options
        self.selected_index = 0
        self.title = title
        self.font_title = pygame.font.Font(None, 60)
        self.font_item = pygame.font.Font(None, 40)
        
    def draw(self):
        self.screen.fill(BG_COLOR)
        title_surf = self.font_title.render(self.title, True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, 100))
        self.screen.blit(title_surf, title_rect)
        
        start_y = 200
        for i, option in enumerate(self.options):
            color = HIGHLIGHT_COLOR if i == self.selected_index else TEXT_COLOR
            text_str = f"> {option} <" if i == self.selected_index else option
            text_surf = self.font_item.render(text_str, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, start_y + i * 50))
            self.screen.blit(text_surf, text_rect)
        pygame.display.flip()

    def handle_input(self, event):
        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif event.key == pygame.K_RETURN:
            return self.options[self.selected_index]
        return None

def get_action_from_keys(pressed_keys, game_name):
    mapping = keys_to_action.get(game_name, {})
    active_keys = [k for k, v in enumerate(pressed_keys) if v]
    possible_actions = []
    for combo, action in mapping.items():
        if all(pressed_keys[k] for k in combo):
            possible_actions.append((len(combo), action))
    if not possible_actions: return 0
    possible_actions.sort(key=lambda x: x[0], reverse=True)
    return possible_actions[0][1]

# --- Main Async Application ---

async def main():
    # 1. Init Pygame FIRST to allow drawing the loading screen
    pygame.init()
    # SCALED is crucial for browsers with high DPI (125% zoom)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
    pygame.display.set_caption("Gymnasium Web Player")
    
    clock = pygame.time.Clock()

    # --- LOADING PHASE ---
    try:
        screen.fill(BG_COLOR)
        draw_text_centered(screen, "Loading Engines...", -20)
        draw_text_centered(screen, "(This may take a moment)", 20, size=24)
        pygame.display.flip()
        await asyncio.sleep(0) # Yield to browser to let it render this frame

        # 2. Delayed Heavy Imports
        # Doing this here ensures we catch import errors on screen
        global gymnasium, ale_py, create_stochasticity_profile
        
        import gymnasium
        import ale_py
        
        try:
            from stori import create_stochasticity_profile
        except ImportError:
            print("Stori module not found. Proceeding without stochasticity.")
            create_stochasticity_profile = None

        # 3. Load Configs
        config_path = "Exp_configs.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                exp_configs = json.load(f)
                print(f"Loaded config from {exp_configs.keys()} keys")
        else:
            exp_configs = {"Breakout": {"Default": {}}} 
            print("Config file not found, using empty default.")

        screen.fill(BG_COLOR)
        draw_text_centered(screen, "Ready!", 0)
        pygame.display.flip()
        await asyncio.sleep(0.5)

    except Exception:
        # FATAL ERROR CATCHER
        # If imports fail (common in WASM for C extensions like ale-py),
        # we print the traceback to the screen so the user knows.
        err_msg = traceback.format_exc()
        print(err_msg) # Print to console too
        while True:
            draw_error_screen(screen, err_msg)
            await asyncio.sleep(0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return

    # --- GAME LOOP ---
    running = True
    state = "MENU_GAME"
    
    allowed_games = [
        ("Breakout", "breakout"),
        ("Boxing", "boxing"),
        ("Gopher", "gopher"),
        ("BankHeist", "bankheist"),
    ]
    game_names_display = [g[0] for g in allowed_games]
    
    menu_game = Menu(screen, game_names_display, "Select Game")
    menu_type = None
    
    selected_game_display = None
    selected_game_internal = None
    selected_type = None
    
    env = None
    stochasticity_profile = None
    episode_score = 0.0  # Track cumulative reward for current episode
    
    # Get service name from environment variable, default to "default"
    service_name = os.environ.get("PLAYER_ID", "default")
    if service_name.isdigit():
        service_name = f"player{service_name}"
    
    # Create scores directory if it doesn't exist
    scores_dir = "scores"
    os.makedirs(scores_dir, exist_ok=True)
    scores_file = os.path.join(scores_dir, f"{service_name}_episode_scores.json")  # JSON file to store scores

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if state == "MENU_GAME":
                    selection = menu_game.handle_input(event)
                    if selection:
                        selected_game_display = selection
                        selected_game_internal = next(g[1] for g in allowed_games if g[0] == selection)
                        types = list(exp_configs.get(selected_game_display, {"Default":{}}).keys())
                        menu_type = Menu(screen, types, f"Select {selection} Config")
                        state = "MENU_TYPE"
                        
                elif state == "MENU_TYPE":
                    if event.key == pygame.K_ESCAPE:
                        state = "MENU_GAME"
                    else:
                        selection = menu_type.handle_input(event)
                        if selection:
                            selected_type = selection
                            state = "INIT_GAME"
                
                elif state == "PLAYING":
                    if event.key == pygame.K_ESCAPE:
                        if env: env.close()
                        env = None
                        state = "MENU_GAME"

        if state == "MENU_GAME":
            menu_game.draw()
            
        elif state == "MENU_TYPE":
            menu_type.draw()
            
        elif state == "INIT_GAME":
            screen.fill(BG_COLOR)
            draw_text_centered(screen, f"Launching {selected_game_display}...", 0)
            pygame.display.flip()
            await asyncio.sleep(0.1)
            
            try:
                config_data = exp_configs.get(selected_game_display, {}).get(selected_type, {})
                
                # Convert flat config to nested structure (as expected by create_stochasticity_profile)
                config_data = nested_dict_from_flat(config_data)
                
                # Determine Stoch Type
                stoch_type = config_data.get('stochasticity_type', None)
                if stoch_type is None:
                    stoch_type = selected_type.split()[-1] if " " in selected_type else selected_type
                
                # Create Profile
                if create_stochasticity_profile:
                    stochasticity_profile = create_stochasticity_profile(
                        game_name=selected_game_internal,
                        type=str(stoch_type),
                        config=config_data
                    )
                else:
                    stochasticity_profile = None

                # Make Env
                env_name = f"ALE/{selected_game_display}-v5"
                gymnasium.register_envs(ale_py)
                
                new_env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
                
                if stochasticity_profile:
                    env = stochasticity_profile.get_env(new_env)
                else:
                    env = new_env
                    
                env.reset()
                episode_score = 0.0  # Reset score when starting new game
                state = "PLAYING"
            except Exception as e:
                # Catch game load errors and show them
                err = traceback.format_exc()
                while True:
                    draw_error_screen(screen, f"Game Load Error:\n{err}\n\nPress ESC to return")
                    await asyncio.sleep(0.1)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            state = "MENU_GAME"
                            break
                    if state == "MENU_GAME": break

        elif state == "PLAYING":
            if env:
                keys = pygame.key.get_pressed()
                action = get_action_from_keys(keys, selected_game_internal)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Accumulate reward for current episode
                episode_score += reward
                
                rgb_array = env.render()
                
                if rgb_array is not None:
                    h, w, c = rgb_array.shape
                    # Transpose if necessary, but typically (H, W, C) -> (W, H)
                    surf = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
                    
                    scale = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    surf = pygame.transform.scale(surf, (new_w, new_h))
                    
                    x_pos = (SCREEN_WIDTH - new_w) // 2
                    y_pos = (SCREEN_HEIGHT - new_h) // 2
                    
                    screen.fill(BG_COLOR)
                    screen.blit(surf, (x_pos, y_pos))
                    pygame.display.flip()
                
                if terminated or truncated:
                    # Store the final score with timestamp before resetting
                    timestamp = datetime.now().isoformat()
                    
                    # Create composite key with game name, subtype, and timestamp
                    game_name = selected_game_display if selected_game_display else "Unknown"
                    subtype = selected_type if selected_type else "No_Type"
                    # Sanitize names for use in JSON keys (replace spaces/special chars)
                    game_name_safe = game_name.replace(" ", "_").replace("/", "_")
                    subtype_safe = subtype.replace(" ", "_").replace("/", "_")
                    score_key = f"{game_name_safe}_{subtype_safe}_{timestamp}"
                    
                    # Load existing scores if file exists
                    scores_data = {}
                    if os.path.exists(scores_file):
                        try:
                            with open(scores_file, "r") as f:
                                scores_data = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            scores_data = {}
                    
                    # Add new score entry
                    scores_data[score_key] = float(episode_score)
                    
                    # Save to JSON file
                    try:
                        with open(scores_file, "w") as f:
                            json.dump(scores_data, f, indent=2)
                        print(f"Saved score: {episode_score} for {game_name} ({subtype}) at {timestamp}")
                    except IOError as e:
                        print(f"Error saving score: {e}")
                    
                    episode_score = 0.0  # Reset for next episode
                    env.reset()

        clock.tick(30)
        await asyncio.sleep(0)

    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())