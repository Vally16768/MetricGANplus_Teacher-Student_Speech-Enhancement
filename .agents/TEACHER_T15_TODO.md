# T15 teacher-improvement TODO

Status: **in progress — validation**
Last update: **2026-07-28**
Next action: **full suite, guard, clean snapshot and CUDA smoke**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T15.0.1 | Close T14 below PESQ gate | select `+0.009365`; auxiliaries safe | passed |
| T15.0.2 | Predeclare OOF calibration | `.agents/TEACHER_T15_PLAN.md` | passed |
| T15.1.1 | Implement OOF prediction and affine shrinkage | nested 5x4 CV; train-only | passed |
| T15.1.2 | Add exact folding/round-trip tests | deterministic and affine equivalence | passed |
| T15.1.3 | Suite, hashes, guard and CUDA smoke | suite `101/101`; guard passed; smoke pending | in-progress |
| T15.2.1 | Full fit/rank/select | no test | blocked |
| T15.2.2 | Apply gate and confirm if passing | unchanged | blocked |
| T15.2.3 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T14 and opened T15 | quadratic route reached `+0.009365`; gap `0.000635` | implement OOF calibration |
| 2026-07-28 | Implemented T15 estimator and campaign flow | nested train-only OOF calibration; targeted `20/20` | run complete validation |
| 2026-07-28 | Passed static validation | full suite `101/101`; project guard `0` issues | commit and run CUDA smoke |
