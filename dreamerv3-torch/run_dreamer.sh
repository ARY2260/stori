# Stochasticity type
# 1: Intrinsic Stochastic Env (action-dependent) - Stochasticity based on agent's random actions.
# 2.1: Intrinsic Stochastic Env (action-independent-random) - Random stochasticity effects.
# 2.2: Intrinsic Stochastic Env (action-independent-concept-drift) - Concept drift over time.
# 3.1: (DEFAULT) Partially observed Env (state-variable-different-repr) - Different state representation.
# 3.2: Partially observed Env (state-variable-missing) - Missing state variables.
stochasticity_type='3.1'

# intrinsic_stochasticity
action_dependent_stochastic_action_prob=0.3

# action_independent_random_stochasticity
action_independent_random_mode='2'
action_independent_random_random_stochasticity_prob=0.25

# action_independent_concept_drift_stochasticity
action_independent_concept_drift_temporal_mode='sudden' # sudden, cyclic
action_independent_concept_drift_temporal_threshold=500
action_independent_concept_drift_secondary_concept_type='3.2'

# partial_observation_stochasticity
partial_observation_type='ram' # blackout, crop, ram
partial_observation_mode='2'
partial_observation_prob=0.7

# seed
seed=7

# game_name
game_name="Boxing"

# run_name
stochasticity_detail="ram_mode2_p0.7"
run_name="atari_${game_name}_stochastic_type${stochasticity_type}_${stochasticity_detail}_seed${seed}"

# logdir
logdir="./logdir/${run_name}"

# run the dreamer
python dreamer.py --configs atari100k \
--seed ${seed} \
--task "atari_${game_name}" \
--logdir ${logdir} \
--stochasticity_type ${stochasticity_type} \
--intrinsic_stochasticity.action_dependent.stochastic_action_prob ${action_dependent_stochastic_action_prob} \
--intrinsic_stochasticity.action_independent_random.mode ${action_independent_random_mode} \
--intrinsic_stochasticity.action_independent_random.random_stochasticity_prob ${action_independent_random_random_stochasticity_prob} \
--intrinsic_stochasticity.action_independent_concept_drift.temporal_mode ${action_independent_concept_drift_temporal_mode} \
--intrinsic_stochasticity.action_independent_concept_drift.temporal_threshold ${action_independent_concept_drift_temporal_threshold} \
--intrinsic_stochasticity.action_independent_concept_drift.secondary_concept_type ${action_independent_concept_drift_secondary_concept_type} \
--partial_observation.type ${partial_observation_type} \
--partial_observation.mode ${partial_observation_mode} \
--partial_observation.prob ${partial_observation_prob}