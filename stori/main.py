from stori.base_stochastic_env_classes import StochasticEnv
from stori.atari_envs import get_breakout_wrapper_registry, get_boxing_wrapper_registry, get_gopher_wrapper_registry, get_bankheist_wrapper_registry
from typing import Dict, Any

# Game registry mapping
GAME_REGISTRIES = {
    'breakout': get_breakout_wrapper_registry,
    'boxing': get_boxing_wrapper_registry,
    'gopher': get_gopher_wrapper_registry,
    'bankheist': get_bankheist_wrapper_registry,
    # Add more games here as needed
}

def create_stochasticity_profile(game_name: str, type: int, config: Dict[str, Any]) -> StochasticEnv:
    """
    Factory function to create StochasticEnv for any supported game.
    
    Args:
        game_name: Name of the game (e.g., 'breakout', 'boxing')
        type: Stochasticity type (0-5)
        config: Configuration dictionary
        
    Returns:
        StochasticEnv instance with game-specific wrappers

    Note:
    Environment types:
        0: Deterministic Env - No stochasticity or partial observability applied.
        1: Intrinsic Stochastic Env (action-dependent) - Stochasticity based on agent's actions.
        2: Intrinsic Stochastic Env (action-independent-random) - Random stochasticity effects.
        3: Intrinsic Stochastic Env (action-independent-concept-drift) - Concept drift over time.
        4: Partially observed Env (state-variable-different-repr) - Different state representation.
        5: Partially observed Env (state-variable-missing) - Missing state variables.
    """
    game_name = game_name.lower()
    if game_name not in GAME_REGISTRIES:
        raise ValueError(f"Unsupported game: {game_name}. Supported games: {list(GAME_REGISTRIES.keys())}")
    
    registry_getter = GAME_REGISTRIES[game_name]
    registry = registry_getter()

    return StochasticEnv(type=type, config=config, wrapper_registry=registry)
