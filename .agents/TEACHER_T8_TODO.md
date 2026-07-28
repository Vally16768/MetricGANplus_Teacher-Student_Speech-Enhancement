# T8 teacher-improvement TODO

Status: **complete negative outcome — oracle ceiling gate**
Last update: **2026-07-28**
Next action: **none; T9 is active**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T8.0.1 | Close T7 below gate | PESQ `+0.004931`; SI-SDR `-0.246202` | passed |
| T8.0.2 | Freeze 256/128 train-only support | T3 train offsets 576/832 | passed |
| T8.0.3 | Predeclare features/ridge/thresholds/gates | `.agents/TEACHER_T8_PLAN.md` | passed |
| T8.1.1 | Implement deployable router | checkpoint config; no clean inference input | passed |
| T8.1.2 | Add feature, selection and round-trip tests | focused suite 11/11 | passed |
| T8.1.3 | Run full suite, guard and clean CUDA smoke | 92/92; guard; `20260728-t8-router-smoke-wb-s3003-a1` | passed |
| T8.2.1 | Generate fit/cal labels and apply oracle/router gates | oracle `+0.014197` below `+0.015`; validation unread | failed |
| T8.2.2 | Evaluate frozen router on rank/select | no test | blocked |
| T8.2.3 | Apply final teacher gate | unchanged thresholds | blocked |
| T8.2.4 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T8.2.5 | Predeclare successor if failed | `.agents/TEACHER_T9_PLAN.md` | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T7 and opened T8 | global and bin-adaptive corrections both exhausted SI-SDR before `+0.01`; per-utterance heterogeneity remains untested | implement adaptive router |
| 2026-07-28 | Implemented T8 feature/ridge/router pipeline | exact base/candidate selection, synthetic ridge recovery and checkpoint round-trip passed 11/11 | full validation |
| 2026-07-28 | Passed pre-runtime validation | 92/92 tests; campaign split validation and project guard passed | commit and clean CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | 10/10 labels, ridge, thresholds, round-trip and pre-validation stop worked; verification-only oracle `+0.014694` is not scientific evidence | production support |
| 2026-07-28 | Completed production support gates | learned `+0.009197` with safe auxiliary deltas, but oracle `+0.014197` missed its frozen gate; no validation/test read | close T8 and activate T9 |
