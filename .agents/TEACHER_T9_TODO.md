# T9 teacher-improvement TODO

Status: **complete negative outcome — auxiliary-risk gate**
Last update: **2026-07-28**
Next action: **none; T10 is active**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T9.0.1 | Close T8 at oracle gate | oracle `+0.014197`; learned `+0.009197` | passed |
| T9.0.2 | Freeze fit/fresh calibration support | 256 train + 128 T3 calibration | passed |
| T9.0.3 | Predeclare actions/ridge/thresholds/gates | `.agents/TEACHER_T9_PLAN.md` | passed |
| T9.1.1 | Implement deployable multi-action router | checkpoint config; no clean inference input | passed |
| T9.1.2 | Add action-selection and round-trip tests | exact action, round-trip, partition/clean disjointness | passed |
| T9.1.3 | Run full suite, guard and clean CUDA smoke | 94/94; guard; `20260728-t9-router-smoke-wb-s3003-a1` | passed |
| T9.2.1 | Generate action labels and apply pre-validation gates | oracle `+0.031116`; learned guardrails failed | failed |
| T9.2.2 | Evaluate frozen router on rank/select | no test | blocked |
| T9.2.3 | Apply final teacher gate | unchanged thresholds | blocked |
| T9.2.4 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T9.2.5 | Predeclare successor if failed | `.agents/TEACHER_T10_PLAN.md`; final gate unchanged | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T8 and opened T9 | learned router generalized but single-action oracle ceiling was insufficient | implement four-action router |
| 2026-07-28 | Implemented T9 model/support/search/CLI | exact four-action selection, portable checkpoint and fresh calibration partition | full validation |
| 2026-07-28 | Passed complete CPU/unit and split validation | 94/94 tests; VoiceBank split audit zero overlap | refresh hashes and guard |
| 2026-07-28 | Passed architecture/privacy/project guard | refreshed source hashes; zero issues | commit and CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | 10/10 fit/cal, four ridges, action selection and checkpoint round-trip; intended pre-validation stop; no validation/test read | production 256/128 search |
| 2026-07-28 | Completed production T9 | oracle `+0.031116`; threshold `0.02` retained `+0.017368` but SI-SDR `-0.279120`; validation/test unread | close T9 and activate T10 |
