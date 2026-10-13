# Engineering record

Initial implementation: both networks use seven learned primitive action logits;
shared observed/experiential features and a learned attention decoder. Parameter
counts matched within 0.2%. Explicit full gradients verified numerically, including
temporal and internal-iteration paths; no framework download is required.

Engineering seeds 29,000,000+ and 31,000,000+ are separate from train/validation/formal.
64-update pilot was insufficient: both variants scored 0 on engineering probe
success. Preserved under engineering/pilot. A separately recorded 1,024-update
pilot scored .4375 flat/varied and .84375 graph/varied on 8 engineering worlds, with
graph independent need success 1.0 and dependency .6875. This is not a formal claim.
Both long pilots are preserved under engineering/pilot_long.

Before formal training, fixed a common budget of 2,048 updates x 16 steps x 24 worlds
per model, with checkpoints 1,024/1,536/2,048 and validation-based selection. Five
initializations for all four cells. Longer-chain formal tests remain unopened.
Training does not contain token-to-token prerequisites.

The formal evaluation details clarify that controlled preparation/inventory reset
precedes a physical survey, then the paired intervention changes only body. They
also explicitly delimit neural dependency processing versus a learned multi-step
future-body world model, which this version does not implement.
