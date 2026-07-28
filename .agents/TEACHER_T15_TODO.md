# T15 teacher-improvement TODO

Status: **in progress — implementation**
Last update: **2026-07-28**
Next action: **implement deterministic OOF calibration and tests**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T15.0.1 | Close T14 below PESQ gate | select `+0.009365`; auxiliaries safe | passed |
| T15.0.2 | Predeclare OOF calibration | `.agents/TEACHER_T15_PLAN.md` | passed |
| T15.1.1 | Implement OOF prediction and affine shrinkage | train-only | in-progress |
| T15.1.2 | Add exact folding/round-trip tests | pending | blocked |
| T15.1.3 | Suite, hashes, guard and CUDA smoke | pending | blocked |
| T15.2.1 | Full fit/rank/select | no test | blocked |
| T15.2.2 | Apply gate and confirm if passing | unchanged | blocked |
| T15.2.3 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T14 and opened T15 | quadratic route reached `+0.009365`; gap `0.000635` | implement OOF calibration |
