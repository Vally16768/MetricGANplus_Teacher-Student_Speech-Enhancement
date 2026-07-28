# T16 teacher-improvement TODO

Status: **in progress — validation**
Last update: **2026-07-28**
Next action: **commit clean snapshot and run CUDA smoke**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T16.0.1 | Close T15 below PESQ gate | select `+0.009070`; auxiliaries safe | passed |
| T16.0.2 | Predeclare fine action set and protocol | `.agents/TEACHER_T16_PLAN.md` | passed |
| T16.1.1 | Generalize quadratic fit/router to eight actions | exact frozen lows | passed |
| T16.1.2 | Add action-count and round-trip tests | exact eight actions/config | passed |
| T16.1.3 | Suite, hashes, guard, clean CUDA smoke | `102/102`; guard; smoke pending | in-progress |
| T16.2.1 | Full fit/rank/select | no test | blocked |
| T16.2.2 | Apply gate and confirm if passing | unchanged | blocked |
| T16.2.3 | Stop campaign if failed | no T17; preserve and audit evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T15 and opened T16 | OOF calibration did not improve T14 | implement fine actions |
| 2026-07-28 | Implemented and statically validated T16 | full suite `102/102`; guard `0` issues | commit and smoke CUDA |
