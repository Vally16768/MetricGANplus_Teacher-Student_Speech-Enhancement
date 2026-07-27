# Iterative execution TODO

Status: **active**
Last evidence update: **2026-07-27**
Current phase: **P4 — teacher-only T1 improvement**
Next action: **P4.7 — strict full-support discriminator calibration**

This is the detailed execution board for the active research sequence. The
campaign-wide summary remains `.agents/TODO.md`; this file owns the subtask
state, dependencies, evidence and next action for P1–P6.

## Agent operating rule

At the start of every project iteration:

1. read this file and inspect the evidence for the single `in-progress` item;
2. reconcile that state with the filesystem, active process and immutable run
   records;
3. execute only the first unblocked item;
4. attach concrete evidence before changing an item to `passed` or `failed`;
5. update `Current phase`, `Next action`, the progress log and
   `.agents/TODO.md`;
6. leave at most one item `in-progress`;
7. never skip a failed or blocked gate by relabeling partial evidence.

Allowed states: `pending`, `in-progress`, `blocked`, `passed`, `failed`,
`not-applicable`.

## Dependency chain

```text
P1 close/audit S0
-> P2 repair and validate resume
-> P3 sanitize/promote S0
-> P4 teacher-only T1
-> teacher gate
   | fail -> stop and report
   | pass -> P5 C1 + fresh S1-WB/S1-NB
-> P6 final comparison/report/audit

TTS remains parked in a separate future campaign.
```

## P1 — Close and audit the converged S0 baseline

Gate: the merged T0→C0→S0 package must reconcile the original epoch-20 run
with its immutable continuation and establish the selected WB/NB checkpoints.

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P1.1 | Confirm continuation completion | no active process; run status `audited`; two cells; `valid_for_promotion=true` | passed |
| P1.2 | Evaluate NB on all frozen splits | `val_select`, `val_rank`, test; NB/8 kHz reference; PESQ-NB metadata and support | passed |
| P1.3 | Verify both selected models and training states | model/training-state hashes, ancestry, selected/best/stop epochs, scheduler/early-stop record | passed |
| P1.4 | Compare epoch 20 with converged checkpoints | same split/protocol comparison; no cross-band score comparison | passed |
| P1.5 | Build merged baseline report, tables and plots | T0/S0-WB/S0-NB metrics, 20→converged deltas, curves, support, limitations | passed |
| P1.6 | Run independent merged-package audit | zero unresolved issues; paths and public/private boundaries reconciled | passed |
| P1.7 | Decide baseline closure gate | valid or failed with cause; update C14/C41 and this board | passed |

Observed immutable continuation evidence:

| Cell | Epoch-20 best | Converged best | Delta | Stop | Interpretation |
|---|---:|---:|---:|---:|---|
| S0-WB / PESQ-WB | 2.596915 @ 20 | 2.602952 @ 34 | +0.006037 | 42 | early stopping; not ceiling-limited |
| S0-NB / PESQ-NB | 3.192184 @ 18 | 3.216751 @ 41 | +0.024567 | 49 | early stopping; not ceiling-limited |

Final continuation metrics already present:

| Cell | `val_select` | `val_rank` | test | Protocol |
|---|---:|---:|---:|---|
| S0-WB | 2.602952 | 2.598257 | 3.051937 | WB reference, PESQ-WB |
| S0-NB | 3.216751 | 3.198166 | 3.615061 | NB reference, PESQ-NB |

Selected artifact hashes:

| Artifact | SHA-256 |
|---|---|
| S0-WB model | `dc1d2d2171876fb5665bd447506e3371492a4619cc8f2749cbfca7292f1ca335` |
| S0-NB model | `1b89e6b5931eb3a4bb63db7844ffe5e74486e9bf75b835342926776336d11491` |
| S0-WB training state | `64dd420df96b0a05d51a6ab923da2e5d3228d28686cb91c3db4155fb27601a36` |
| S0-NB training state | `dd74c67625b374d2f8fe0d56f3e32cf74b6b4f5bcae959c64ebceffb3d85d592` |

Continuation evidence root:
`local/runs/20260727-official-students-cont50-s0-a1` (private/ignored).
Its two-cell audit reports zero issues, two models and 36 report samples. The
separate merged baseline/continuation closure audit is recorded below.

Closure evidence root:
`local/runs/20260727-converged-s0-baseline-a1` (private/ignored). The
`close-baseline` command bound all three source model hashes, generated the
comparison CSV/report/convergence plot and independently reconciled 3/3 cells,
3/3 models and 54 reported samples with zero issues.

Ceiling rule: if a future NB best occurs at the configured maximum epoch, mark
it `ceiling-limited`, do not claim convergence and do not extend the ceiling
without a separate analysis and decision. The current NB best is epoch 41 and
stopped at 49 through early stopping, so this rule is not triggered.

## P2 — Repair resume robustness

Gate: interrupted and uninterrupted training must produce identical
post-evaluation control state and selected checkpoint under a deterministic
fixture.

This repair does not invalidate the completed S0 continuation because that run
finished normally.

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P2.1 | Reproduce and localize the resume-state defect | focused failing test or state-order trace | passed |
| P2.2 | Save scheduler and early-stopping state after every evaluation | minimal training-loop change; architecture/training docs updated | passed |
| P2.3 | Add interrupt/resume equivalence test | LR, bad-epoch/patience count, best epoch/score/hash and next action match uninterrupted control | passed |
| P2.4 | Run focused and full unit suites | all tests pass in shared venv | passed |
| P2.5 | Run project guard and canonical config validation | zero guard issues; plan/config validation passes | passed |
| P2.6 | Run clean real-entry-point CUDA resume smoke | interrupted/resumed package reconciles; no dataset mutation | passed |
| P2.7 | Commit and push verified repair | clean snapshot and recorded commit | passed |

Unblock condition satisfied: P1.7 passed.

## P3 — Sanitize and promote the valid S0 baseline

Gate: only selected, audited and portable artifacts enter Git.

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P3.1 | Build the promotion inventory | selected WB/NB weights, metrics, plots, config, report and hashes only | passed |
| P3.2 | Sanitize public provenance | no personal path, username, host, mount, dataset location or server logic | passed |
| P3.3 | Verify exclusions | no dataset, teacher cache, generated audio, replay or regenerable bulk | passed |
| P3.4 | Validate promoted run contract | canonical run validation and independent metric/artifact reconciliation | passed |
| P3.5 | Update `.agents`, README and documentation index | no stale baseline claims or duplicated source of truth | passed |
| P3.6 | Run tests and project guard | required gates pass from the promotion snapshot | passed |
| P3.7 | Commit and push baseline release | public commit/hash recorded; Git worktree clean | passed |
| P3.8 | Correct variable-length evaluation discovered in P4 | per-utterance inference; re-evaluated T0/S0 package; v1 marked superseded; v2 audit/push | passed |

Unblock condition: P1 and P2 pass.

## P4 — Teacher-only T1 improvement

Detailed method: `.agents/TEACHER_IMPROVEMENT_PLAN.md`.

Gate: T1 must improve true `val_select` PESQ-WB by at least `+0.01`, with
STOI loss at most `0.002` and SI-SDR loss at most `0.25 dB`. Test is
reporting-only.

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P4.1 | Freeze E0 official-teacher reference | official checkpoint ancestry and WB protocol | passed |
| P4.2 | Implement calibration-only teacher command | no G update; at least 100 current outputs/refresh | passed |
| P4.3 | Implement calibration guard | held-out current-output MAE/correlation/range; failed guard skips G | passed |
| P4.4 | Complete resume-state tests for G/D loop | G, D, optimizers, scheduler, patience, replay and history restore | passed |
| P4.5 | Run unit/integration tests and project guard | all required gates pass | passed |
| P4.6 | Run clean CUDA smoke | current/history/current, clean=1, true noisy/enhanced PESQ, local replay observed | passed |
| P4.7 | Run strict calibration, then monitored teacher-only pilot | full-support D gate first; E0/E1/E2 only after pass; immutable runs; no cache/S1 | in-progress |
| P4.8 | Audit teacher gate | true metrics and calibration reconcile independently | pending |
| P4.9 | Record gate decision | pass selects T1; fail stops downstream work | pending |

Unblock condition: P3.7 passes.

## P5 — Build C1 and train fresh S1 students

Run only if P4.9 passes.

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P5.1 | Build content-addressed C1 | accepted T1 hash; frozen manifest; Desktop-local FP16; no input audio copies | blocked |
| P5.2 | Train S1-WB from zero | S0-matched architecture, seed and max-50/plateau/early-stop policy | blocked |
| P5.3 | Train S1-NB from zero | same controlled policy; NB reference and PESQ-NB | blocked |
| P5.4 | Apply convergence rule | ceiling-limited handling without automatic extension | blocked |
| P5.5 | Audit C1/S1 package | ancestry, hashes, histories, support and bandwidth protocols reconcile | blocked |

Unblock condition: accepted T1 teacher.

## P6 — Final comparison and research report

| ID | Item | Required evidence | Status |
|---|---|---|---|
| P6.1 | Compare T1−T0 | WB true metrics with uncertainty and fixed support | blocked |
| P6.2 | Compare S1-WB−S0-WB | PESQ-WB and WB guard metrics | blocked |
| P6.3 | Compare S1-NB−S0-NB | PESQ-NB and NB guard metrics | blocked |
| P6.4 | Generate final tables/figures/report | claim-to-artifact traceability | blocked |
| P6.5 | Run independent promotion audit | zero unresolved issues or explicit failed result | blocked |
| P6.6 | Commit and push final accepted evidence | sanitized canonical package only | blocked |

## Parked — TTS metric-critic study

The TTS direction is not part of P1–P6. It starts only after the enhancement
line is stable and audited, with a separate generator, dataset, discriminator
calibration, evaluation protocol, provenance and claim set.

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-27 | Created the iterative board after NB completion | continuation status `audited`; audit zero issues; WB best 34/stop 42; NB best 41/stop 49 | P1.5 merged baseline report |
| 2026-07-27 | Bound the board to the project skill and validated the control plane | skill validator passed; 48/48 tests passed; project guard reported 0 issues | P1.5 merged baseline report |
| 2026-07-27 | Closed the converged S0 baseline | `20260727-converged-s0-baseline-a1`; 3 cells/models, 54 samples, zero audit issues | P2 resume robustness |
| 2026-07-27 | Localized and repaired post-evaluation resume state ordering | focused interrupted/resumed control test matches LR, patience, best state and history | P2.4 full suite |
| 2026-07-27 | Passed static resume-repair validation | 51/51 tests; canonical config/plan valid; project guard zero issues | P2.6 clean CUDA smoke |
| 2026-07-27 | Resume smoke A1 stopped before its injected interruption | PyTorch rejected NumPy RNG state encoded as `torch.uint32`; failed run preserved | encode RNG as serializable `int64`, test, rerun A2 |
| 2026-07-27 | Resume smoke A2 reached final reconciliation but exposed CUDA kernel variance | control/resumed PESQ differed by about `6e-6` while LR/patience/best epoch aligned; exact model hashes differed | deterministic CUDA fault-injection smoke A3 |
| 2026-07-27 | Resume smoke A3 stopped on an unsupported deterministic CUDA kernel | `reflection_pad1d_backward` has no deterministic CUDA implementation; changing the frontend would invalidate S0 | CUDA forward/backward with optimizer effect frozen for exact state-equivalence A4 |
| 2026-07-27 | Resume smoke A4 passed exact state equivalence | real CUDA forward/backward; injected stop after epoch 2; resumed/control LR, patience, best state, selected hash and history identical; optimizer effect isolated because the CUDA reflection-pad backward is nondeterministic | P2.7 commit/push |
| 2026-07-27 | Closed P2 on pushed commit `5c48415` | 52/52 tests, plan/config validation and project guard passed; resume smoke A4 audit has zero issues | P3.1 promotion inventory |
| 2026-07-27 | Built and audited the portable S0 package | `20260727-converged-s0-baseline-v1`; 3 models, 23 inventoried artifacts, zero package/privacy issues; 56/56 tests, contract/plan/guard pass | P3.7 commit/push |
| 2026-07-27 | Published the converged S0 baseline | Git commit `e6388d4`; 24 package files in Git normal, push confirmed and worktree clean | P4.1 freeze E0 |
| 2026-07-27 | Implemented teacher-only calibration and trial flows | E0 hash/protocol binding; disjoint D/held-out current outputs; failed-gate G skip; D optimizer/replay resume; 60/60 tests and config/plan validation pass | P4.6 clean CUDA smoke |
| 2026-07-27 | Passed clean teacher-only CUDA smoke on `8eb21ff` | calibration and E0/E1/E2 packages audit with zero issues; current/history/current, 2+2 disjoint FP16 outputs, no cached inputs, one calibrated G update | P4.7 strict calibration |
| 2026-07-27 | Stopped strict calibration before D after detecting batch-sensitive E0 | same checkpoint/manifest produced PESQ 2.7126 at batch 4 versus 2.6989 at batch 8 because BLSTM consumed right-padding; stopped run preserved | P3.8 true-length evaluation correction |
| 2026-07-27 | Published corrected padding-invariant S0 baseline v2 | exact three model hashes retained; true-length run and public package audits zero issues; 61/61 tests, plan, guard, contract and privacy checks passed; commit `65b9a9c` pushed | P4.7 strict discriminator calibration |
| 2026-07-27 | First strict full-support calibration failed safely | A2 audited with zero package issues; 100 held-out, normalized MAE 0.1968, Pearson 0.5379, Spearman 0.5504, range failed; zero G updates | predeclared second and final D refresh attempt |
| 2026-07-27 | Validated the fixed-generator two-refresh retry flow | 62/62 tests; plan/config/guard pass; CUDA smoke has two refreshes, zero G updates, no redundant epoch/final evaluation and zero audit issues | commit/push, then strict A3 |
