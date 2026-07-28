# T7 teacher-improvement TODO

Status: **complete negative outcome — below teacher gate**
Last update: **2026-07-28**
Next action: **none; T8 is active**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T7.0.1 | Close T6 below gate | exact T5 selection; PESQ `+0.005075` | passed |
| T7.0.2 | Freeze fresh disjoint 96/96 support | T3 train offsets 384/480 | passed |
| T7.0.3 | Predeclare 24 confidence candidates | 8-fit/4-cal/2-rank funnel | passed |
| T7.1.1 | Implement confidence-conditioned logits | checkpoint-configured, no clean inference input | passed |
| T7.1.2 | Add parity, formula and checkpoint round-trip tests | focused suite 9/9 | passed |
| T7.1.3 | Run full suite, guard and clean CUDA smoke | 90/90; guard; `20260728-t7-confidence-smoke-wb-s3003-a1` | passed |
| T7.2.1 | Run full true-PESQ funnel | `20260728-t7-confidence-wb-s3003-a1`; no test | passed |
| T7.2.2 | Apply unchanged teacher gate | PESQ `+0.004931`; STOI `-0.001245`; SI-SDR `-0.246202` | failed |
| T7.2.3 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T7.2.4 | Predeclare successor if failed | `.agents/TEACHER_T8_PLAN.md` | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T6 and opened T7 | T6 scale `1.0` reproduced T5 exactly; global temperature added no capacity | implement confidence-conditioned logits |
| 2026-07-28 | Implemented deployable T7 transform and search funnel | disabled parity, exact formula, 24-candidate grid and checkpoint round-trip tests passed 9/9 | full validation |
| 2026-07-28 | Passed pre-runtime validation | 90/90 tests; campaign split validation and project guard passed | commit and clean CUDA smoke |
| 2026-07-28 | Completed clean contracted CUDA smoke | two candidates, fit/cal/rank/select flow, T0 fallback and checkpoint round-trip completed; test unread | production search |
| 2026-07-28 | Completed production T7 | selected `low=-0.30/high=0/threshold=0`; safe `+0.004931` PESQ but below gate; test unread | close T7 and activate T8 |
