# T5 teacher-improvement TODO

Status: **in progress — predeclared method and implementation**
Last update: **2026-07-28**
Current phase: **T5.1 implementation and validation**
Next action: **commit the validated T5 snapshot and run contracted CUDA smoke**

Allowed states: `pending`, `in-progress`, `blocked`, `passed`, `failed`,
`not-applicable`.

## T5.0 — Cause and protocol

| ID | Item | Evidence | Status |
|---|---|---|---|
| T5.0.1 | Reconcile terminal T4 evidence | T4-A `+0.002034`; T4-B `-0.000001`; no promoted teacher | passed |
| T5.0.2 | Review primary black-box/PESQ evidence | MetricGAN+, stable black-box SE and PESQ exploitation risk | passed |
| T5.0.3 | Freeze train/cal/rank/select roles | 96/96 T3-train identities; untouched `val_rank`/`val_select`; no test | passed |
| T5.0.4 | Predeclare curve/search/bounds/gate | 8 knots; bounds `[-.20,.05]`; steps `.08/.04/.02`; unchanged gate | passed |
| T5.0.5 | Index T5 as active source of truth | index/skill/TODO/documentation synchronized | passed |

## T5.1 — Implementation and validation

| ID | Item | Evidence | Status |
|---|---|---|---|
| T5.1.1 | Implement smooth bias-curve folding | exact 257-bin ordinary checkpoint | passed |
| T5.1.2 | Implement deterministic coordinate search | true PESQ; fit-only coordinate decisions; sweep calibration | passed |
| T5.1.3 | Add unit/provenance/split tests | curve bounds/equivalence/disjoint roles | passed |
| T5.1.4 | Run full tests, validators and guard | 87/87; plan/campaign valid; guard/diff pass | passed |
| T5.1.5 | Run clean contracted CUDA smoke | nonpromotable two-file path | pending |

## T5.2 — Scientific run and exit

| ID | Item | Evidence | Status |
|---|---|---|---|
| T5.2.1 | Run contracted full search | 48 fit candidates; 3 calibrated sweeps; rank candidates | pending |
| T5.2.2 | Apply one-shot `val_select` gate | gain `>=.01`; STOI/SI-SDR guards; no test | pending |
| T5.2.3 | Independently re-evaluate passed checkpoint | exact metrics/hash/support provenance | blocked |
| T5.2.4 | Confirm across three declared seeds/bootstrap | positive mean; paired PESQ CI excludes zero | blocked |
| T5.2.5 | Promote evidence and shut down after success | immediate power check; no active writes | blocked |
| T5.2.6 | Predeclare capacity successor if T5 fails | preserve negative evidence; no gate relaxation | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Opened T5 after terminal T4-B result | surrogate parameter gradient regresses with horizon; selected delta effectively zero | freeze true-PESQ low-dimensional method |
| 2026-07-28 | Predeclared eight-knot zeroth-order search | separated 96/96 train support; three fixed coordinate steps; hard quality guards | implement and test |
| 2026-07-28 | Implemented T5 fit/cal/rank/select flow | eight-knot ordinary checkpoint, fit-only coordinate decisions, disjoint support and one-shot select | complete validation |
| 2026-07-28 | Passed complete T5 pre-run validation | 87/87 tests; real campaign/research plan; architecture hashes; guard zero issues | clean commit and CUDA smoke |
