import ale_py
import gymnasium
import stori

gymnasium.register_envs(ale_py)

stochasticity_config = {
                        # Select stochasticity type from:
                        # 1: Intrinsic action-dependent stochastic env - Stochasticity based on agent's actions.
                        # 2.1: Intrinsic action-independent random stochastic env - Random stochasticity effects.
                        # 2.2: Intrinsic action-independent concept drift stochastic env - Concept drift over time.
                        # 3.1: Partially observed env - Different state representation. (Default ALE env is type 3.1)
                        # 3.2: Partially observed env - Missing state variables.
                        'stochasticity_type': '3.1',

                        'intrinsic_stochasticity': {

                            # Intrinsic action-dependent stochasticity parameters
                            'action_dependent': {
                                'stochastic_action_prob': 0.5 # probability of applying the modification (0.0 - 1.0)
                                },

                            # Intrinsic action-independent concept drift stochasticity parameters
                            'action_independent_concept_drift': {
                                'temporal_threshold': 300, # Steps after which the concept drifts occurs
                                'temporal_mode': 'cyclic', # Select mode from: 'cyclic', 'sudden'
                                'secondary_concept_type': '3.2', # Select secondary concept type from: '1', '2.1', '3.2'
                                },

                            # Intrinsic action-independent random stochasticity parameters
                            'action_independent_random': {
                                'mode': '3', # modes are game specific
                                'random_stochasticity_prob': 0.25, # probability of applying the modification (0.0 - 1.0)
                                }
                            },

                        # Partially observed env - Missing state variables stochasticity parameters
                        'partial_observation': {
                            'type': 'ram', # Select type from: 'ram', 'crop', 'blackout'
                            'mode': '4', # modes are game specific
                            'prob': 0.75, # probability of applying the modification (0.0 - 1.0)
                            },
                        }

available_games = [
                "Breakout",
                "Boxing",
                "Gopher",
                "BankHeist",
                ]

game_name = available_games[1]

# Initialize ALE environment
# Note: make sure to keep frameskip=1 and repeat_action_probability=0.0 to use STORI stochasticity profile
env = gymnasium.make(f"ALE/{game_name}-v5", full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)

# Initialize STORI stochasticity profile
stochasticity_profile = stori.create_stochasticity_profile(game_name.lower(), stochasticity_config['stochasticity_type'], config=stochasticity_config)
stochastic_env = stochasticity_profile.get_env(env)

# Basic testing
print(f"Basic testing - {game_name}")
obs, _ = stochastic_env.reset()

for i in range(1000):
    action = stochastic_env.action_space.sample()
    obs, reward, terminated, truncated, info = stochastic_env.step(action)
    if i % 100 == 0:
        print("obs.shape:", obs.shape,
            "reward:", reward,
            "terminated:", terminated,
            "truncated:", truncated,
            "info:", info)
    if terminated or truncated:
        obs, _ = stochastic_env.reset()

stochastic_env.close()
