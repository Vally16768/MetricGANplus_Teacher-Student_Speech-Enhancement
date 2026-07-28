# T11 teacher-improvement TODO

Status: **complete — negative outcome**
Last update: **2026-07-28**
Next action: **continue from `.agents/TEACHER_T12_TODO.md`**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T11.0.1 | Close T10 below PESQ gate | select `+0.008015`; auxiliary guards pass | passed |
| T11.0.2 | Freeze remaining fresh support | final 72 T3 audit identities | passed |
| T11.0.3 | Predeclare penalty/margin grid and gate | `.agents/TEACHER_T11_PLAN.md` | passed |
| T11.1.1 | Implement penalized deployable router | bias-folded penalty; no clean inference input | passed |
| T11.1.2 | Add support/policy/round-trip tests | penalty immutability plus inherited exact round-trip | passed |
| T11.1.3 | Run suite, guard and CUDA smoke | 96/96; guard; `20260728-t11-router-smoke-wb-s3003-a1` | passed |
| T11.2.1 | Run fresh 72-example calibration | `+0.014979` PESQ; auxiliaries pass | passed |
| T11.2.2 | Evaluate rank/select conditionally | rank `+0.010068`; select `+0.008349`; test unread | passed |
| T11.2.3 | Apply final teacher gate | PESQ below `+0.01` | failed |
| T11.2.4 | Independently confirm/promote/shutdown | not applicable after failed gate | blocked |
| T11.2.5 | Predeclare successor if failed | `.agents/TEACHER_T12_PLAN.md` | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T10 and opened T11 | auxiliary-safe T10 missed PESQ gate by `0.001985`; strongest action dominated | implement strength-penalized routing |
| 2026-07-28 | Implemented T11 support/search/CLI | frozen ridges, bias-folded penalty, fresh remaining audit support | complete validation |
| 2026-07-28 | Passed complete unit/integration suite | 96/96 tests | refresh hashes and guard |
| 2026-07-28 | Passed split/configuration and project guard | zero issues | commit and CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | 10-row fresh support, 25 policies, checkpoint and intended prevalidation stop | production 72-row search |
| 2026-07-28 | Completed production T11 | run `20260728-t11-router-wb-s3003-a1`; calibration/rank passed, select PESQ `+0.008349`; test unread | open T12 |
