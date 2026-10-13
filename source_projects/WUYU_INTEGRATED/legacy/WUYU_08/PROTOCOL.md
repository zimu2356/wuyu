# WUYU_08: need-sensitive causal composition

Protocol written before implementation/training. This is a bounded mechanistic
experiment, not a claim of consciousness, life, or a completed foundation model.
WUYU_07 remains unchanged. The six-site physics, experiential associative memory,
and serialization are reused with attribution; motor and planning networks are new.

## Main question and 2 x 2 comparison

Can training coverage and a relational iterative neural architecture make current
body needs control use of individually acquired knowledge?

Four cells: flat/regular, flat/varied, graph/regular, graph/varied. Regular uses
ordinary WUYU_07-like continuous life; varied adds exogenous body deficits during
training. Both see the same relation permutations across births and all ordinary
phases. No training example contains the additional token-to-token dependency
reserved for longer-chain tests. Architectures share raw observation, learned
per-site facts, derived pairwise dependency links, teacher information boundary,
primitive seven-action interface, and differentiable action decoder. Parameter
counts are approximately matched; recurrent state sizes and compute are reported.
Both receive primitive-action, ultimate-site and first-subgoal teacher supervision.
Thus regular is an improved/common-control training recipe, not an exact rerun of
the original 07 action-only student. Comparisons isolate architecture bundles and
coverage, not message iteration alone. Iteration ablations are additional probes.

Each neural controller emits primitive actions. No teacher, recursive route finder,
or hand-written navigation/execution policy is called at deployment. Engineered
input encodings, ring distances, dependency-link construction, and basic needs are
declared priors. Auxiliary ultimate-goal output alone is not proof of goal use.

## Data and selection

Five independent training initializations 801..805. Same seed schedule across cells.
Train seeds start 8,100,000 + initialization_index * 1,000,000 + birth * batch;
validation seeds 19,000,000..; formal seeds 20,000,000..; engineering >=29,000,000.
Exact batch, update, checkpoint and learning-rate settings will be frozen in CONFIG
after a separately labelled engineering pilot and before any formal results.
All candidates and selection decisions retained. No extra training or selection
after opening formal evaluation. Validation selection prioritizes the mean of
independent-need and single-dependency first-needed-resource success, then action
agreement on validation observations; never selects by formal results.

## Evaluation

1. Common-history probes: all models receive exactly the same physical teacher
   history, and form their own memories solely from those observations/actions.
   The teacher may guide this acquisition history, explicitly not autonomous
   learning. A physical survey makes current local prerequisites observable.
   Matched clones swap only body deficits (.25,.9) / (.9,.25). Position, prep,
   inventory, recurrent state and prior history otherwise retained. The observer
   may change world laws but never fills the agent's memory from simulator truth.
2. Four relation conditions: independent devices, ordinary new dependency, held-out
   token-to-token longer chain, changed body-device prerequisites. Frozen plastic
   model/motor parameters in the main probes isolate reuse; local observation of
   prerequisites and transient recurrent state continue. Online-learning variants
   are reported separately where used.
3. Each probe runs a full 40 physical actions unless dead, retaining raw steps.
   Report first-needed-body-resource success, any-needed-resource success, first
   resource identity, completion cost, body change, and predicted goal/target.
   A first wrong resource stays a primary failure even if later recovered.
4. Autonomous continuous life: 1,536 events, own learned memory and sampled neural
   actions, no guided warm-up. Four original physical phases plus fixed external
   resource-loss events to expose need changes without resetting history. Report
   survival-time fraction, resources and adaptation, not infinite survival.
5. Same-history controller interventions: body input clamp; recurrent hidden-state
   reset; graph iteration budget reduction; graph dependency-message removal; and
   planner reading the same initial experiential memory as a diagnostic comparator.
   Interventions may be distribution shifts and are not standalone causal proof.

Formal 32 world seeds x 5 fixed trained initializations. Pair by world and average
initializations/need directions before a world bootstrap. Intervals are conditional
on these trained models. Also report each initialization. Do not count correlated
branches as independent worlds or merge all gates into a 'life score'.

## Decision rules

Report all four cells and paired coverage/architecture effects, whether positive,
zero or negative. Mechanistic support requires correct need-sensitive choices plus
successful closed-loop execution on unseen worlds. Longer-chain performance is an
explicit generalization limit; do not hide it inside overall averages. Improved
metrics with changed data do not establish a new architecture. Shared policy
weights remain fixed in formal life; individual associative/motor parameters are
plastic. Open-ended concepts, value-origin learning and shared-weight lifelong
plasticity remain future work, not claimed by this experiment.

## Verification and traceability

Gradient finite differences for both architectures and recurrent/iterative paths;
hidden-state persistence; death/energy/prerequisite physics; no privileged input;
no runtime teacher calls; frozen-probe parameters; no probe writeback; exact resume
of actual trained checkpoints; independent replay of raw physics and recomputation
of metrics. Record source/config hashes before formal and verify unchanged afterward.
Retain failed runs, engineering fixes and negative results.
