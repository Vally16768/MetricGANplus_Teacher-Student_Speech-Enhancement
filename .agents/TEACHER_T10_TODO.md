# T10 teacher-improvement TODO

Status: **in progress — production fresh calibration**
Last update: **2026-07-28**
Next action: **commit smoke evidence and run the 128-example T10 search**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T10.0.1 | Close T9 at auxiliary-risk gate | oracle `+0.031116`; threshold `0.02` SI-SDR `-0.279120` | passed |
| T10.0.2 | Freeze T9 regressors and fresh support | first 128 T3 audit identities | passed |
| T10.0.3 | Predeclare margins and unchanged final gate | `.agents/TEACHER_T10_PLAN.md` | passed |
| T10.1.1 | Implement risk-margin calibration | no refit; exact T0 fallback | passed |
| T10.1.2 | Add support/selection/round-trip tests | fresh support test plus inherited exact T9 selection/round-trip | passed |
| T10.1.3 | Run suite, guard and clean CUDA smoke | 95/95; guard; `20260728-t10-router-smoke-wb-s3003-a1` | passed |
| T10.2.1 | Run fresh 128-example calibration | stop before validation on failure | in-progress |
| T10.2.2 | Evaluate rank/select conditionally | no test | blocked |
| T10.2.3 | Apply final teacher gate | unchanged thresholds | blocked |
| T10.2.4 | Independently confirm/promote/shutdown | multi-seed + bootstrap + audit | blocked |
| T10.2.5 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T9 and opened T10 | strong oracle and PESQ gain, but PESQ-only decisions over-consumed SI-SDR | implement conservative margin calibration |
| 2026-07-28 | Implemented T10 support/search/CLI | frozen T9 regressors, fresh T3-audit support, exact T0 fallback and conditional validation | complete validation |
| 2026-07-28 | Passed complete unit/integration suite | 95/95 tests | refresh hashes and guard |
| 2026-07-28 | Passed split/configuration and project guard | zero overlap; zero guard issues | commit and CUDA smoke |
| 2026-07-28 | Completed clean CUDA smoke | fresh 10-row support, margin selection, rank/select flow and round-trip passed; verification-only | production fresh calibration |
