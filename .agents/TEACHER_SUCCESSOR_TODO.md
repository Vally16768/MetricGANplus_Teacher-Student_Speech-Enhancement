# T2 teacher successor TODO

Status: **active — no training started**  
Last update: **2026-07-28**  
Current phase: **T2.3 D2-OFFICIAL**
Next action: **implement resumable D-only fitting and gates**

This board owns the iterative execution of
`TEACHER_SUCCESSOR_PLAN.md`. The completed P1–P6 board and failed T1 evidence
remain unchanged.

## Operating rule

At every iteration:

1. reconcile this board with Git state, processes and immutable run evidence;
2. execute the first unblocked item only;
3. keep at most one item `in-progress`;
4. attach tests, hashes, metrics or audit evidence before changing status;
5. stop at every failed gate; never relabel a failed run;
6. update `.agents/TODO.md`, this board and the documentation index together;
7. run training only on CUDA from the shared Desktop virtual environment;
8. commit/push only a clean, tested and explicitly authorized snapshot.

Allowed states: `pending`, `in-progress`, `blocked`, `passed`, `failed`,
`not-applicable`.

## Dependency flow

```text
T2.1 official parity
-> T2.2 fixed D support
-> T2.3 D2-OFFICIAL calibration
   | pass ------------------------+
   | coverage/ranking fail        |
   v                              |
 T2.4 D2-RANGE                    |
   | pass ------------------------+
   | fail -> stop                 |
                                  v
                         T2.5 E1/E2 pilot
                                  |
                       teacher promotion gate
                       | fail -> stop/report
                       | pass
                       v
                     T2.6 three-seed confirmation
                                  |
                                  v
                     T2.7 C2 + fresh S2-WB/NB
                                  |
                                  v
                     T2.8 final audit/report
```

## T2.1 — Exact official parity

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.1.1 | Pin official SpeechBrain revision and recipe files | revision, file hashes, provenance record | passed |
| T2.1.2 | Test frontend and normalized-label parity | fixed tensor/PESQ fixtures within tolerance | passed |
| T2.1.3 | Test architecture/state-dict output parity | same weights produce matching outputs | passed |
| T2.1.4 | Test batch-1 update and replay-order parity | clean/enh/noisy, current/history/current trace | passed |
| T2.1.5 | Test true-length and batch invariance | no padding/batch-induced score change | passed |
| T2.1.6 | Run full tests, plan validation and project guard | all pass, zero guard issues | passed |

Gate: all parity and invariance checks pass before any D2 fitting.

## T2.2 — Fixed discriminator support

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.2.1 | Freeze D train/calibration/audit IDs | pair/clean-utterance disjoint; source speaker IDs unavailable | passed |
| T2.2.2 | Generate T0 candidates locally | T0/manifest hashes, FP16, no input copies | passed |
| T2.2.3 | Compute true PESQ-WB labels | finite labels, mode/sample-rate provenance | passed |
| T2.2.4 | Audit score and condition coverage | plots/tables by type, score and estimated SNR | passed |
| T2.2.5 | Verify dataset read-only boundary | zero source mutation; local cache only | passed |

Unblock: T2.1 passes.

## T2.3 — `D2-OFFICIAL`

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.3.1 | Implement resumable D-only fitting | D/optimizer/scheduler/patience/replay state | in-progress |
| T2.3.2 | Add interrupted/resumed equivalence test | selected hash and control state match | pending |
| T2.3.3 | Run clean CUDA smoke | batch 1, three passes, true labels observed | pending |
| T2.3.4 | Fit D2 to declared stopping rule | immutable history and selected checkpoint | pending |
| T2.3.5 | Apply untouched audit fidelity gate | MAE/correlation/range/subgroups | pending |
| T2.3.6 | Apply local directional gate | sign agreement and delta-rank report | pending |
| T2.3.7 | Record D2 decision | pass or exact failure cause | pending |

Unblock: T2.2 passes.

## T2.4 — Conditional `D2-RANGE`

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.4.1 | Confirm eligible D2 failure mode | coverage/local-rank failure; parity intact | blocked |
| T2.4.2 | Build declared train-only score widening | interpolation/perturbation manifest | blocked |
| T2.4.3 | Refit with identical audit/stopping protocol | immutable run and checkpoint | blocked |
| T2.4.4 | Reapply full and local gates | independent audit report | blocked |
| T2.4.5 | Record final discriminator decision | accepted D2 or downstream stop | blocked |

Unblock only if T2.3 fails for the predeclared eligible reason. If T2.3
passes, mark T2.4 `not-applicable`.

## T2.5 — Matched teacher pilot

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.5.1 | Freeze E0 and matched E1/E2 configs | same init/seed/support/schedule | blocked |
| T2.5.2 | Test rejected-update rollback and resume | exact G/D/control-state restoration | blocked |
| T2.5.3 | Run E1 control CUDA smoke | no metric gradient | blocked |
| T2.5.4 | Run E2 metric CUDA smoke | accepted D only; guards active | blocked |
| T2.5.5 | Run monitored E1/E2 pilot | `val_rank` histories; immutable outputs | blocked |
| T2.5.6 | Evaluate candidate pair on `val_select` | paired PESQ/STOI/SI-SDR | blocked |
| T2.5.7 | Decide optional E3 LR | only safe positive/sub-threshold E2 | blocked |
| T2.5.8 | Record pilot gate | pass, fail or inconclusive | blocked |

Unblock: one D2 branch passes both discriminator gates.

## T2.6 — Three-seed teacher confirmation

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.6.1 | Run declared E0/E1/E2 seed set | complete immutable runs | blocked |
| T2.6.2 | Compute paired uncertainty | seed and utterance-level intervals | blocked |
| T2.6.3 | Apply teacher promotion gate | +0.01 PESQ, STOI/SI-SDR guards, E2>E1 | blocked |
| T2.6.4 | Evaluate selected T2 on test once | reporting-only test record | blocked |
| T2.6.5 | Audit and select teacher hash | zero unresolved provenance/metric issues | blocked |

Unblock: T2.5 passes.

## T2.7 — Transfer accepted T2 to students

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.7.1 | Build content-addressed C2 | accepted T2 hash; local FP16 outputs | blocked |
| T2.7.2 | Train fresh S2-WB | S0-matched architecture/policy; PESQ-WB | blocked |
| T2.7.3 | Train fresh S2-NB | S0-matched architecture/policy; PESQ-NB | blocked |
| T2.7.4 | Apply max-50 convergence rule | early stop or ceiling-limited decision | blocked |
| T2.7.5 | Audit C2/S2 provenance | hashes, histories, support and bandwidth | blocked |

Unblock: T2.6 teacher promotion passes.

## T2.8 — Final evidence and promotion

| ID | Item | Evidence | Status |
|---|---|---|---|
| T2.8.1 | Compare `T2−T0` | paired WB teacher metrics | blocked |
| T2.8.2 | Compare `S2-WB−S0-WB` | WB reference/PESQ-WB | blocked |
| T2.8.3 | Compare `S2-NB−S0-NB` | NB reference/PESQ-NB | blocked |
| T2.8.4 | Generate article-ready report/figures | claim-to-artifact map | blocked |
| T2.8.5 | Run independent package/privacy audit | zero unresolved issues | blocked |
| T2.8.6 | Promote authorized Git artifacts | clean tested commit/push | blocked |

Unblock: T2.7 passes. If an upstream gate fails, replace downstream items with
`not-applicable` and publish the negative evidence instead.

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Predeclared T2 after final T1 calibration failure | T1 A3: MAE/correlation/range failed, zero G updates; official recipe comparison completed | T2.1 exact official parity |
| 2026-07-28 | Activated the clean T2 successor sequence | one worktree/main; no active training; GPU available; plan/guard valid | pin official SpeechBrain revision |
| 2026-07-28 | Passed exact SpeechBrain v1.1.0 D parity | pinned `36c180c`; corrected magnitude/constant-pad/time-first frontend; 65/65 tests; CUDA D/G-gradient smoke; plan/campaign/guard pass | freeze D2 support IDs |
| 2026-07-28 | Prepared and independently audited fixed D2 support A3 | 1000/200/200; 1400 unique; support `545ac1bf`; source hashes unchanged; no missing/non-FP16/nonfinite records; T0 PESQ mean 2.861 | implement resumable D-only fitting |
