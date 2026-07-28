# T7 teacher-improvement TODO

Status: **in progress — clean CUDA smoke**
Last update: **2026-07-28**
Next action: **commit validated T7 and run contracted CUDA smoke**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T7.0.1 | Close T6 below gate | exact T5 selection; PESQ `+0.005075` | passed |
| T7.0.2 | Freeze fresh disjoint 96/96 support | T3 train offsets 384/480 | passed |
| T7.0.3 | Predeclare 24 confidence candidates | 8-fit/4-cal/2-rank funnel | passed |
| T7.1.1 | Implement confidence-conditioned logits | checkpoint-configured, no clean inference input | passed |
| T7.1.2 | Add parity, formula and checkpoint round-trip tests | focused suite 9/9 | passed |
| T7.1.3 | Run full suite, guard and clean CUDA smoke | 90/90 + validate + guard pass; smoke pending | in-progress |
| T7.2.1 | Run full true-PESQ funnel | fit/cal/rank/select; no test | blocked |
| T7.2.2 | Apply unchanged teacher gate | PESQ/STOI/SI-SDR | blocked |
| T7.2.3 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T7.2.4 | Predeclare successor if failed | preserve evidence; no relaxed gate | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T6 and opened T7 | T6 scale `1.0` reproduced T5 exactly; global temperature added no capacity | implement confidence-conditioned logits |
| 2026-07-28 | Implemented deployable T7 transform and search funnel | disabled parity, exact formula, 24-candidate grid and checkpoint round-trip tests passed 9/9 | full validation |
| 2026-07-28 | Passed pre-runtime validation | 90/90 tests; campaign split validation and project guard passed | commit and clean CUDA smoke |
