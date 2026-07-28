# T14 teacher-improvement TODO

Status: **in progress — implementation validation**
Last update: **2026-07-28**
Next action: **run full tests, campaign validation, hashes and guard**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T14.0.1 | Close T13 below PESQ gate | rank `+0.010932`; select `+0.008806`; auxiliaries safe | passed |
| T14.0.2 | Predeclare quadratic successor | `.agents/TEACHER_T14_PLAN.md` | passed |
| T14.1.1 | Implement deterministic 152-feature transform | checkpoint-configured/noisy-only | passed |
| T14.1.2 | Implement regularized metric-delta fits and policy flow | 584 train-only; 336 rank policies | passed |
| T14.1.3 | Tests, hashes, guard and CUDA smoke | first smoke found constructor forwarding defect; fixed; 99/99 | in-progress |
| T14.2.1 | Full fit/rank/select | no test | blocked |
| T14.2.2 | Apply final teacher gate | unchanged | blocked |
| T14.2.3 | Independently confirm/promote/shutdown | only after candidate pass | blocked |
| T14.2.4 | Predeclare successor if failed | preserve evidence | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Closed T13 and opened T14 | best routed select `+0.008806`; linear fit remains limited | add regularized interactions |
| 2026-07-28 | Implemented quadratic model/checkpoint/search path | 152 deployable features; quadratic metric deltas | validate implementation |
| 2026-07-28 | Passed full suite and campaign validation | 99/99; VoiceBank split audit valid | hashes and guard |
| 2026-07-28 | Refreshed hashes and passed project guard | zero issues | commit and CUDA smoke |
| 2026-07-28 | First smoke stopped at checkpoint reload | transform was saved but omitted by intermediate constructors; select/test unread | fix forwarding and retest |
| 2026-07-28 | Fixed all constructor forwarding and added real round-trip regression | focused plus full 99/99 pass | guard, commit and repeat smoke |
