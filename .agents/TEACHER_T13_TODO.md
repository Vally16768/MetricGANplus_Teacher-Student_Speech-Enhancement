# T13 teacher-improvement TODO

Status: **complete — negative outcome**
Last update: **2026-07-28**
Next action: **continue from `.agents/TEACHER_T14_TODO.md`**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T13.0.1 | Close T12 below PESQ gate | rank `+0.011176`; select `+0.008425`; auxiliaries safe | passed |
| T13.0.2 | Predeclare multi-objective policy family | `.agents/TEACHER_T13_PLAN.md` | passed |
| T13.1.1 | Implement train-only metric-delta fits | 584 refs; PESQ/STOI/SI-SDR ridge per action | passed |
| T13.1.2 | Implement exact deployable utility folding | no metric/reference at inference | passed |
| T13.1.3 | Add tests, hashes, guard and CUDA smoke | 98/98; guard; `20260728-t13-router-smoke-wb-s3003-a1` | passed |
| T13.2.1 | Fit full support and rank 336 policies | rank `+0.010932`; auxiliaries safe | passed |
| T13.2.2 | Evaluate one checkpoint on `val_select` | select `+0.008806`; test unread | passed |
| T13.2.3 | Apply final gate | PESQ below `+0.01` | failed |
| T13.2.4 | Independently confirm/promote/shutdown | only after candidate pass | blocked |
| T13.2.5 | Predeclare successor if failed | `.agents/TEACHER_T14_PLAN.md` | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T12 and opened T13 | T12 failed PESQ by `0.001575` with auxiliaries safe | replace global risk with predicted per-action auxiliary deltas |
| 2026-07-28 | Implemented T13 support, fits, policy grid and campaign flow | train-only labels; exact folded inference; single final split read | validate implementation |
| 2026-07-28 | Passed unit/integration suite and split validation | 98/98; VoiceBank campaign valid | refresh hashes and guard |
| 2026-07-28 | Refreshed architecture hashes and passed project guard | zero issues | commit and run CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | fit/rank labels, 12 ridges, 336 policies and checkpoint round-trip; intended small-support stop | production search |
| 2026-07-28 | Completed production T13 | `20260728-t13-router-wb-s3003-a1`; select `+0.008806`; auxiliaries safe | open T14 |
