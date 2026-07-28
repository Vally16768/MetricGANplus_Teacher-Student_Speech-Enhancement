# T13 teacher-improvement TODO

Status: **in progress — implementation validation**
Last update: **2026-07-28**
Next action: **run tests, campaign validation, architecture hashes and guard**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T13.0.1 | Close T12 below PESQ gate | rank `+0.011176`; select `+0.008425`; auxiliaries safe | passed |
| T13.0.2 | Predeclare multi-objective policy family | `.agents/TEACHER_T13_PLAN.md` | passed |
| T13.1.1 | Implement train-only metric-delta fits | 584 refs; PESQ/STOI/SI-SDR ridge per action | passed |
| T13.1.2 | Implement exact deployable utility folding | no metric/reference at inference | passed |
| T13.1.3 | Add tests, hashes, guard and CUDA smoke | 98/98; campaign/guard pass; smoke pending | in-progress |
| T13.2.1 | Fit full support and rank 336 policies | `val_rank` only | blocked |
| T13.2.2 | Evaluate one checkpoint on `val_select` | no test | blocked |
| T13.2.3 | Apply final gate | unchanged | blocked |
| T13.2.4 | Independently confirm/promote/shutdown | only after candidate pass | blocked |
| T13.2.5 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T12 and opened T13 | T12 failed PESQ by `0.001575` with auxiliaries safe | replace global risk with predicted per-action auxiliary deltas |
| 2026-07-28 | Implemented T13 support, fits, policy grid and campaign flow | train-only labels; exact folded inference; single final split read | validate implementation |
| 2026-07-28 | Passed unit/integration suite and split validation | 98/98; VoiceBank campaign valid | refresh hashes and guard |
| 2026-07-28 | Refreshed architecture hashes and passed project guard | zero issues | commit and run CUDA smoke |
