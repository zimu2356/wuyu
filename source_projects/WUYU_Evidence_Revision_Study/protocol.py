"""Pre-development commitments. No holdout-dependent choices are permitted."""
from copy import deepcopy

METHODS = ['B0', 'BE', 'BU', 'BEU', 'BR']
CASES = ['stable', 'local_swap', 'opportunity', 'noise_pulse', 'visible_shock', 'recurrence']
CONFIG = dict(
    version=1, life_ticks=2400, post_window=512,
    development_seeds=dict(start=61000, count=32), pilot_seeds=dict(start=60000, count=8),
    holdout_seed_start=71000, holdout_count_candidates=[64,128],
    precision_target=.01, stable_precision_target=.001,
    minimum_useful_effect=.02, stable_noninferiority_margin=.002,
    bootstrap_resamples=10000, bootstrap_seed=917031,
    fatigue_penalty=.08, action_cost=.002, exploration_weight=.15,
    model_window=24, ridge=.002, recency_cap=.04, recency_horizon=240,
    residual_threshold=.04, residual_streak=3, residual_cosine=.5,
    detector_min_samples=8, revision_old_weight=.1, revision_cooldown=6,
    random_probe_probability=.05,
    body_drift=[.004,.004], body_relaxation=.003,
    noise_sd=.003, noisy_sd=.06, noise_duration=128,
    wait_recovery=.12, work_fatigue=.045, material_period=80,
    training_materials=[.2,.8], change_low=800, change_high=1200,
    recurrence_gap=600, diagnostic_sampling_actions=16, diagnostic_waits_per_probe=3,
    diagnostic_followup=128, causal_followup=64,
    methods=METHODS, cases=CASES,
    calibration_rule=(
        'Fixed E and U parameters; no optimization for candidate victory. Pilot checks physics, '
        'oracle feasibility, stable cost and runtime. BR probability becomes the clipped [0.005,0.15] '
        'mean stable-pilot fraction of BE decisions changed by the E term. '
        'Then run all 32 development seeds once. N is the smallest of [64,128] satisfying BOTH '
        '1.96*paired-SD/sqrt(N)<=0.01 for primary local_swap contrast and <=0.001 for stable '
        'contrast. If neither satisfies both, use 128 and disclose precision shortfall. '
        'No sample extension or parameter changes after formal freeze.'),
    primary='Mean body squared loss over actions tau through tau+511: BE minus BEU in local_swap.',
    success_rule='Primary paired-seed percentile 95% lower bound >0.02 AND stable-life BEU-minus-BE upper bound <0.002.',
    stable_endpoint='Mean body squared loss over ALL 2400 stable-world post-action observations.',
    recovery='First 64-step postchange window mean <= max(0.02, 2*prechange256 mean); must also hold for next 64 steps. Search 512 steps; failures right-censored at 512.',
    prediction_accuracy_latency='After each changed-axis tool observation, first time both changed-tool body-delta MSE <0.0025 at body [0.2,0.2], fatigue .25 and each material .2/.8. Evaluator-only, first crossing not persistence; cap 512.',
    multiple_comparisons='One confirmatory main contrast plus stable guard; all other intervals descriptive, not multiplicity-adjusted.',
    unit='One independent base seed; methods and cases paired within seed. All failures retained.',
    boundaries='Given goals, features, actions, detector and revalidation incentive; no formed desires, value standards, skills, social relationships or subjective experience.',
)

def fresh_config():
    return deepcopy(CONFIG)
