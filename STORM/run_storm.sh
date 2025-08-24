env_name=Boxing
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -stochasticity_config_path "config_files/configs_stochastic.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path " " \
    --'stochasticity_type' 2 \
    --intrinsic_stochasticity.action_dependent.stochastic_action_prob 0.3 \
    --intrinsic_stochasticity.action_independent_random.mode '2' \
    --intrinsic_stochasticity.action_independent_random.random_stochasticity_prob 0.25 \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_mode 'sudden' \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_threshold 300 \
    --intrinsic_stochasticity.action_independent_concept_drift.secondary_concept_type 5 \
    --partial_observation.type 'blackout' \
    --partial_observation.mode '9' \
    --partial_observation.prob 1.0

python -u eval.py \
    -env_name "ALE/${env_name}-v5" \
    -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
    -config_path "config_files/STORM.yaml" \
    -stochasticity_config_path "config_files/configs_stochastic.yaml" \
    --'stochasticity_type' 2 \
    --intrinsic_stochasticity.action_dependent.stochastic_action_prob 0.3 \
    --intrinsic_stochasticity.action_independent_random.mode '2' \
    --intrinsic_stochasticity.action_independent_random.random_stochasticity_prob 0.25 \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_mode 'sudden' \
    --intrinsic_stochasticity.action_independent_concept_drift.temporal_threshold 300 \
    --intrinsic_stochasticity.action_independent_concept_drift.secondary_concept_type 5 \
    --partial_observation.type 'blackout' \
    --partial_observation.mode '9' \
    --partial_observation.prob 1.0
