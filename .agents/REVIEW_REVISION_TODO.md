# IEEE review revision execution board

Status: **in progress — runtime validation**
Last update: **2026-07-30**
Next action: **project guard, clean commit and CUDA smoke for cached/non-cached ablations**

This board addresses the reviewer assessment attached to
`MetricGAN_Teacher_Student_IEEE_Draft_v2.pdf`. It is a student-evidence
campaign and does not reopen the closed T1--T16 teacher search.

| ID | Review item | Required evidence | Status |
|---|---|---|---|
| R0 | Reconcile draft claims and canonical S0 package | draft/review/S0 audit; response ledger | passed |
| R1 | Current-protocol noisy baselines | WB and NB, 824 test pairs, matched PESQ | pending |
| R2 | Isolate knowledge distillation | WB seed 0: clean-only, teacher-wave only, ERB-mask only, teacher-wave+mask, complete D1 | in-progress |
| R3 | Narrowband references | NB clean-only student and matched-input 8-kHz teacher adapter | pending |
| R4 | Correct lookahead | contract corrected to 10 ms; numerical streaming/dependency test pending | in-progress |
| R5 | Multi-seed evidence | complete D1 WB/NB seeds 0, 1001, 2002 | pending |
| R6 | Uncertainty | sample-level test metrics and paired 95% bootstrap CIs | pending |
| R7 | Exact provenance | teacher/code/config/manifest/speaker identifiers and hashes | pending |
| R8 | Complexity | teacher neural-core MAC/s plus frontend/buffer/activation caveats | pending |
| R9 | References and presentation | foundational citations, new diagram/plots, shorter historical/refinement text | pending |
| R10 | Addressed-review document | every point linked to evidence and article-ready replacement text | pending |
| R11 | Independent audit | tests, guard, run/package/claim reconciliation | pending |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-30 | Audited reviewer text against six-page draft and promoted S0 | `docs/ADDRESSED_REVIEW.md` | implement frozen ablations |
| 2026-07-30 | Added configurable D1 component weights and review run entry point | focused tests; full suite `104/104` | guard and CUDA smoke |
| 2026-07-30 | Corrected mathematical lookahead metadata | `win_length/(2*sample_rate)=10 ms` | add numerical streaming evidence |
| 2026-07-30 | Stopped first clean-only smoke as a harness failure | full 9,754-row train manifest with smoke batch 2; no result claimed | bound verification-only training support |
| 2026-07-30 | Stopped bounded-train smoke A2 after it exposed full validation evaluation | 32 train rows passed; legacy smoke evaluated 128/1,690 rows twice | bound verification-only evaluation support |
| 2026-07-30 | Smoke A3 caught sample-metric API wiring error | bounded train passed; stopped on first two-row rank evaluation; no test | correct function signature and add regression |
| 2026-07-30 | Baseline smoke A1 caught evaluator adapter mismatch | passthrough lacked required true-length `denoise_single`; stopped before metrics | implement/test adapter interface |
| 2026-07-30 | Baseline smoke A2 caught teacher shape mismatch | noisy baselines completed two rows; NB teacher stopped before metrics | preserve evaluator's `(batch, length)` single-waveform contract |

## Frozen experiment matrix

All new training uses the identical causal-max student architecture, optimizer,
VoiceBank+DEMAND splits, maximum-50/plateau/early-stopping policy and
bandwidth-matched evaluation used by canonical S0.

| Cell | Bandwidth | Seed(s) | Mask | Teacher waveform | Clean waveform |
|---|---:|---:|---:|---:|---:|
| A-CLEAN | WB | 0 | 0 | 0 | 1 |
| A-TWAVE | WB | 0 | 0 | 1 | 0 |
| A-MASK | WB | 0 | 1 | 0 | 0 |
| A-TWAVE-MASK | WB | 0 | 12/17 | 5/17 | 0 |
| A-COMPLETE | WB | 0, 1001, 2002 | 0.60 | 0.25 | 0.15 |
| N-CLEAN | NB | 0 | 0 | 0 | 1 |
| N-COMPLETE | NB | 0, 1001, 2002 | 0.60 | 0.25 | 0.15 |

Seed 0 complete cells reuse the promoted S0 checkpoints. New run directories
are immutable and unique. Test is never used for checkpoint selection.

## Exit rule

An item is addressed only when its raw evidence, aggregate output and
article-ready change agree. Unrun listening tests or target-device
measurements remain explicit limitations, not implied accomplishments.
