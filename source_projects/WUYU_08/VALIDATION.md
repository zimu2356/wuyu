# WUYU_08 verification

- 9 core checks passed; see tests.log and tests/test_core.py.
- Independently replayed 1,228,800 autonomous-life event slots,
  550,400 probe event slots and 61,440
  common-history event slots. Physics and probe metrics recomputed.
- All 215 branch records retain initial state, actions, outcomes and final state.
  Shared initial facts match between the four controllers; plastic probe parameters
  and shared network weights remained unchanged where required.
- Four actual trained initialization-0 checkpoints resumed 384 further events across
  32 individuals each, matching every saved next-state field exactly.
- All 20 shared networks stayed fixed through formal life; trained runtime continued
  with teacher/planner functions patched to raise on any call.
- All 385 result NPZ files passed ZIP CRC checks; frozen training and
  formal source hashes match. Reports/audits/packaging are separate from that frozen core.
- Report: 8 tables, four embedded PNGs, mobile viewport, 1,536 recorded
  events; figures visually inspected and JavaScript syntax checked with Node.
  No full-browser pixel render was performed.
- Added failure analysis after seeing the first formal longer-chain failure is
  explicitly exploratory, reuses formal worlds, and does not alter training,
  selection or primary metrics. See SUPPLEMENT_DIAGNOSTIC.md.
- Package script checks archive CRC and every included member SHA-256 and executes
  the light package demo after extraction. Receipts are supplied beside the ZIPs.

The ledger independently expresses physical transitions and observations, but seed
mapping/world construction is reused. Bootstrap intervals condition on five fixed
trained initializations and resample 32 worlds; multiple contrasts are exploratory.
