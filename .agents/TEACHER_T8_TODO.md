# T8 teacher-improvement TODO

Status: **in progress — clean CUDA smoke**
Last update: **2026-07-28**
Next action: **commit validated T8 and run contracted CUDA smoke**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T8.0.1 | Close T7 below gate | PESQ `+0.004931`; SI-SDR `-0.246202` | passed |
| T8.0.2 | Freeze 256/128 train-only support | T3 train offsets 576/832 | passed |
| T8.0.3 | Predeclare features/ridge/thresholds/gates | `.agents/TEACHER_T8_PLAN.md` | passed |
| T8.1.1 | Implement deployable router | checkpoint config; no clean inference input | passed |
| T8.1.2 | Add feature, selection and round-trip tests | focused suite 11/11 | passed |
| T8.1.3 | Run full suite, guard and clean CUDA smoke | 92/92 + validate + guard pass; smoke pending | in-progress |
| T8.2.1 | Generate fit/cal labels and apply oracle/router gates | stop before validation on failure | blocked |
| T8.2.2 | Evaluate frozen router on rank/select | no test | blocked |
| T8.2.3 | Apply final teacher gate | unchanged thresholds | blocked |
| T8.2.4 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T8.2.5 | Predeclare successor if failed | preserve evidence; no relaxed gate | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T7 and opened T8 | global and bin-adaptive corrections both exhausted SI-SDR before `+0.01`; per-utterance heterogeneity remains untested | implement adaptive router |
| 2026-07-28 | Implemented T8 feature/ridge/router pipeline | exact base/candidate selection, synthetic ridge recovery and checkpoint round-trip passed 11/11 | full validation |
| 2026-07-28 | Passed pre-runtime validation | 92/92 tests; campaign split validation and project guard passed | commit and clean CUDA smoke |
