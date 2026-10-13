# Evaluation decisions fixed before formal access

The 2x2 comparison uses shared target/goal supervision (weights .6/.3) plus primitive
action cross entropy (weight 1). Both have the same learned soft-attention motor
decoder. Flat: 11,325 parameters; graph: 11,345, with 3 internal rounds. Flat recurrent
state 30 scalars; graph 240. Equal parameter count is not equal state or compute.
The relational architecture includes object factorization and graph input biases;
a difference between architectures cannot be assigned solely to iteration.

All teacher labels come from each student's own current experiential memory. The
first half of training uses teacher trajectories, then a fixed declining mixture,
and the last quarter entirely student trajectories. The shared teacher has engineered
need weights and recursive planning. This transfers a designed control prior; it
does not demonstrate autonomous invention of the planning algorithm or basic values.
Both coverage conditions already encounter body-device dependencies. Varied adds
body deficits every 48 physical events, not new chain lengths. The coverage factor
therefore chiefly manipulates need coverage. Longer-chain transfer is held out.

Common-history probes use an actual 384-event exploratory teacher history. Each model
replays its observations and executed actions, reconstructing its own hidden state
and memories. No simulator products/codes are inserted into those memories. For each
condition, inventory expiry and preparation reset precede a physical six-step survey.
This survey exposes prerequisites one station at a time. The paired intervention
then changes only body levels, retaining all other post-survey state. It is a
controlled counterfactual assay; autonomous continuous life is assessed separately.
Long-chain test: all body producers consume token 1, its producer consumes token 0;
token 0 producer has no input. Changed test flips body-producer prerequisites.
Both are observable during the same survey, with no phase label given to the agent.

Common-history first-needed-resource success over 40 greedy actions is primary.
Additional scores retain first wrong outcomes, any needed outcome later, cost and
body improvement. Parameter-frozen probes still update currently observed prerequisite
facts and recurrent state. Main life uses sampled actions, plastic personal memories,
five model initializations and 32 paired worlds, plus six fixed external loss events.
No automatic revival. Parent death persists into own-history probes.

All four cells receive body-clamp probes for independent/dependency conditions.
Graph/varied also gets 1-round, 0-round and no-edge probes on ordinary/long dependency;
hidden reset on ordinary dependency; same-memory frozen-parameter planner on all four.
Actual own-life event-384 states get independent/dependency probes in all four cells.
Graph/varied personal-learning-disabled continuous life is an extra ablation.

Future-body trajectories are not explicitly learned/rolled out in this version.
The neural core iteratively processes learned dependency relations and is trained on
goal/subgoal/action decisions. Its shared weights remain fixed in formal individual
life. Online associative knowledge is still allocated to six known sites. These
boundaries must be stated in the report even when behavioral scores improve.
