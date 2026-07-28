# T6 teacher-improvement TODO

Status: **in progress — affine-logit implementation**
Last update: **2026-07-28**
Next action: **commit validated T6 and run clean contracted CUDA smoke**

| ID | Item | Evidence | Status |
|---|---|---|---|
| T6.0.1 | Close T5 below gate | `val_select` PESQ `+0.005075`; SI-SDR `-0.245701` | passed |
| T6.0.2 | Freeze new disjoint 96/96 support | T3 train offsets 192/288 | passed |
| T6.0.3 | Predeclare two curves × seven scales | 14 fit, top-5 cal, top-3 rank | passed |
| T6.1.1 | Implement exact affine folding | ordinary checkpoint | passed |
| T6.1.2 | Add equivalence/split/selection tests | 88/88; exact weight/bias algebra | passed |
| T6.1.3 | Run clean contracted CUDA smoke | verification-only | pending |
| T6.2.1 | Run full true-PESQ grid | fit/cal/rank/select; no test | pending |
| T6.2.2 | Apply teacher gate | unchanged thresholds | pending |
| T6.2.3 | Independently confirm/promote/shutdown | 3 seeds + bootstrap + audit | blocked |
| T6.2.4 | Predeclare successor if failed | preserve evidence; no relaxed gate | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Opened T6 after bounded T5 gain | additive curve generalized but exhausted SI-SDR margin before +0.01 PESQ | implement affine capacity |
| 2026-07-28 | Implemented and validated affine search | fresh support offsets; 14/5/3 funnel; 88/88 tests and guard pass | clean CUDA smoke |
