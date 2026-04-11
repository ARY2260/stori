# uv pip install ".[atari]"
OMP_NUM_THREADS=1 xvfb-run -a uv run python cleanrl_ppo_atari.py \
    --env_name "ALE/BankHeist-v5" \
    --exp_name "BankHeist-stochastic_type3.2_10M_seed0" \
    --total_timesteps 10000000 \
    --capture_video \
    --seed 0 \
    --'stochasticity_type' '3.2' \
    --intrinsic_stochasticity.action_dependent.stochastic_action_prob 0.3 \
    --intrinsic_stochasticity.action_independent_random.mode '3' \
    --intrinsic_stochasticity.action_independent_random.random_stochasticity_prob 0.001 \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_mode 'cyclic' \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_threshold 600 \
    --intrinsic_stochasticity.action_independent_concept_drift.secondary_concept_type '3.2' \
    --partial_observation.type 'ram' \
    --partial_observation.mode '3' \
    --partial_observation.prob 0.75
