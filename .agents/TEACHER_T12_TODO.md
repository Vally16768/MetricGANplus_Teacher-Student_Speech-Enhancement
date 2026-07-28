# T12 teacher-improvement TODO

Status: **in progress — production rank selection**
Last update: **2026-07-28**
Next action: **commit smoke evidence and run the full 128-example rank search**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T12.0.1 | Close T11 below PESQ gate | select `+0.008349`; auxiliary guards pass; test unread | passed |
| T12.0.2 | Predeclare rank-selected policy family | `.agents/TEACHER_T12_PLAN.md` | passed |
| T12.1.1 | Implement one-pass rank policy selection | frozen T9 model/ridges; 72 fixed policies | passed |
| T12.1.2 | Add selection and checkpoint tests | 97/97 tests; inherited round-trip | passed |
| T12.1.3 | Run suite, guard and CUDA smoke | 97/97; guard; `20260728-t12-router-smoke-wb-s3003-a1` | passed |
| T12.2.1 | Run full `val_rank` selection | exact action metrics; no select during ranking | in-progress |
| T12.2.2 | Evaluate one frozen policy on `val_select` | no test | blocked |
| T12.2.3 | Apply final teacher gate | unchanged | blocked |
| T12.2.4 | Independently confirm/promote/shutdown | recompute + confirmations + bootstrap + audit | blocked |
| T12.2.5 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T11 and opened T12 | T11 rank `+0.010068`, select `+0.008349`; auxiliaries safe | implement rank-selected policy |
| 2026-07-28 | Implemented T12 policy grid and campaign entry points | one exact `val_rank` action pass; one conditional `val_select` evaluation; test unread | validate code |
| 2026-07-28 | Passed full unit/integration suite and campaign validation | 97/97; VoiceBank split audit valid | refresh hashes and guard |
| 2026-07-28 | Refreshed architecture hashes and passed project guard | zero issues | commit clean snapshot and run CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | 10 rank rows, 72 policies, safe selection and checkpoint round-trip; select/test unread | production rank search |
