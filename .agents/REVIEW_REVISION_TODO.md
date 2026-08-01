# IEEE review revision execution board

Status: **blocked — validated N-CLEAN loader fix awaits a clean snapshot**
Last update: **2026-08-01**
Next action: **obtain explicit commit authorization, commit the validated fix and registers, then launch a new immutable N-CLEAN run from the clean snapshot**

This board addresses the reviewer assessment attached to
`MetricGAN_Teacher_Student_IEEE_Draft_v2.pdf`. It is a student-evidence
campaign and does not reopen the closed T1--T16 teacher search.

| ID | Review item | Required evidence | Status |
|---|---|---|---|
| R0 | Reconcile draft claims and canonical S0 package | draft/review/S0 audit; response ledger | passed |
| R1 | Current-protocol noisy baselines | WB and NB, 824 test pairs, matched PESQ | passed |
| R2 | Isolate knowledge distillation | WB seed 0: clean-only, teacher-wave only, ERB-mask only, teacher-wave+mask, complete D1 | in-progress |
| R3 | Narrowband references | NB clean-only student and matched-input 8-kHz teacher adapter | in-progress |
| R4 | Correct lookahead | contract corrected to 10 ms; numerical streaming/dependency test pending | in-progress |
| R5 | Multi-seed evidence | additional WB/NB seeds explicitly removed from the active scope by the user | open limitation |
| R6 | Uncertainty | sample-level test metrics and paired 95% bootstrap CIs; no across-seed estimate | in-progress |
| R7 | Exact provenance | teacher/code/config/manifest/speaker identifiers and hashes | passed |
| R8 | Complexity | teacher neural-core MAC/s plus frontend/buffer/activation caveats | in-progress |
| R9 | References and presentation | foundational citations, new diagram/plots, shorter historical/refinement text | in-progress |
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
| 2026-07-30 | Baseline smoke A3 caught cross-band autocast mismatch | noisy baselines completed; FP16 resample reached teacher complex STFT | force the cross-band reference path to FP32 |
| 2026-07-30 | Baseline smoke A4 passed | three systems x two test rows; WB/NB metric modes matched; test not used for selection | launch the predeclared full reporting-only evaluation |
| 2026-07-30 | Full current-protocol baseline evaluation passed independent reconciliation | 824 rows each for `NOISY-WB`, `NOISY-NB` and `MATCHED-INPUT-TEACHER-NB`; CSV hashes and PESQ/STOI/SI-SDR/delta-SNR means reconciled within `1e-6`; 110/110 tests and project guard passed | launch A-CLEAN WB seed-0 |
| 2026-07-30 | Full clean-only WB seed-0 launched from clean commit `f306a97` | `20260730-review-clean-wb-s0-a1`; 9,754 train pairs, no teacher cache, CUDA, test reserved until selection | monitor training and reconcile the completed package |
| 2026-07-30 | Added article-ready provenance, complexity and foundational-reference material | canonical commit/checkpoint/config/manifest hashes; validation speaker IDs; teacher 118.02M neural-core MAC/s derivation; verified DOI list | retain complexity as analytical evidence and complete figure/plot revisions |
| 2026-07-30 | Removed four additional complete-model seed runs from scope by explicit user decision | no WB/NB seeds 1001 or 2002 will be launched; promoted seed-0 complete models remain the comparison rows | report single-seed training variability as an unresolved limitation |
| 2026-07-31 | Reconciled completed clean-only WB seed-0 package | `20260730-review-clean-wb-s0-a1`; 50 epochs, selected epoch 45; 824-pair WB test PESQ 2.5517, STOI 0.9350, SI-SDR 17.9958 dB; result hash and restricted checkpoint metadata match; test read only after selection; 110/110 tests, research-plan validation and project guard passed | produce sample-level paired uncertainty with the final matrix; do not yet make a causal article claim |
| 2026-07-31 | Launched teacher-waveform-only WB seed-0 from clean commit `fbf27c9` | `20260731-review-twave-wb-s0-a1`; official 9,754-row FP16 teacher cache reconciled with zero missing targets; CUDA training active | monitor training and audit the selected-checkpoint package before A-MASK |
| 2026-07-31 | Reconciled completed teacher-waveform-only WB seed-0 package | `20260731-review-twave-wb-s0-a1`; early stop at epoch 40, selected epoch 32; 824-pair WB test PESQ 3.0389, STOI 0.9278, SI-SDR 8.7718 dB; result hash, history and restricted checkpoint metadata match; test read only after selection; 110/110 tests, research-plan validation and project guard passed | retain as aggregate evidence pending sample-level paired uncertainty |
| 2026-07-31 | Launched ERB-mask-only WB seed-0 from clean detached worktree at `fbf27c9` | `20260731-review-mask-wb-s0-a1`; exact weights `[1,0,0]`, official 9,754-row FP16 teacher cache and CUDA training active; primary documentation edits preserved outside the clean runner | monitor training and audit the selected-checkpoint package before A-TWAVE-MASK |
| 2026-07-31 | Paused A-MASK after the user-requested epoch boundary | epoch 37 evaluation completed and history/state persisted at global step 45,103; best remains epoch 31 with `val_select` PESQ-WB 2.595268; process group `SIGSTOP` verified and GPU utilization is zero; private `provenance/pause.json` records the exact resume action | resume in-memory at epoch 38 on user request |
| 2026-07-31 | Classified the paused A-MASK attempt as failed after the host reboot | `20260731-review-mask-wb-s0-a1`; shutdown converted the clean epoch-37 boundary into an interrupted partial-epoch state at global step 45,452, so it is not canonical resume evidence | preserve the failure record and restart from scratch |
| 2026-07-31 | Stopped the first fresh A-MASK retry as an infrastructure failure | `20260731-review-mask-wb-s0-a2`; the detached external VoiceBank input was unavailable before the first optimizer step; status failed at epoch/global step `0/0` | restore and validate the read-only dataset before another immutable retry |
| 2026-07-31 | Launched fresh A-MASK retry after restoring the external dataset read-only | `20260731-review-mask-wb-s0-a3`; clean commit `fbf27c9`, exact weights `[1,0,0]`, all frozen manifest/cache hashes matched, all referenced inputs were readable, split audit and project guard passed; CUDA training reached epoch 1 | monitor to early stop or the 50-epoch ceiling, then independently audit before A-TWAVE-MASK |
| 2026-08-01 | Reconciled completed ERB-mask-only WB seed-0 package | `20260731-review-mask-wb-s0-a3`; early stop at epoch 39, selected epoch 31; 824-pair WB test PESQ 3.0529, STOI 0.9288, SI-SDR 8.6783 dB; result/checkpoint hashes, 39-row history, source contracts and restricted checkpoint metadata match; test read only after selection; 110/110 tests, research-plan validation and project guard passed | retain as aggregate evidence pending sample-level paired uncertainty |
| 2026-08-01 | Launched and monitored teacher-waveform-plus-mask WB seed-0 from clean detached worktree at `fbf27c9` | `20260801-review-twave-mask-wb-s0-a1`; exact weights `[12/17,5/17,0]`, frozen manifest/cache hashes and read-only dataset; epoch 42 final evaluation active, current best epoch 34 with `val_select` PESQ-WB 2.608825; CUDA process healthy and no logged errors | allow final evaluation to finish, then independently audit before N-CLEAN |
| 2026-08-01 | Reconciled completed teacher-waveform-plus-mask WB seed-0 package | `20260801-review-twave-mask-wb-s0-a1`; early stop at epoch 42, selected epoch 34; 824-pair WB test PESQ 3.056460, STOI 0.929191, SI-SDR 8.717301 dB and delta-SNR -0.463088 dB; campaign/checkpoint hashes, 42-row history, source contracts and restricted checkpoint metadata independently reconcile; test read only after selection; 110/110 tests, research-plan validation and project guard passed | retain as aggregate evidence pending sample-level paired uncertainty |
| 2026-08-01 | Preserved first N-CLEAN attempt as an implementation failure | `20260801-review-clean-nb-s0-a1`; default DataLoader collation failed at epoch/global step `0/0`, before any optimizer update or scientific result, because native 16-kHz frame counts were mixed with 8-kHz crop coordinates | correct target-rate segment handling and never reuse this run directory |
| 2026-08-01 | Corrected and validated N-CLEAN target-rate crop/pad behavior | source lengths are converted to the requested sample-rate coordinates and every loaded pair is finally cropped/padded to the exact target-rate segment length; focused short/long NB, default-collation and WB-preservation tests pass, as do eight real 9,754-row-manifest NB batches with four workers, the complete 113/113 suite, the real `campaign.py validate` entry point, research-plan validation, architecture hashes and project guard | obtain explicit authorization for the required clean fix commit, then use a new immutable N-CLEAN run ID |

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
| A-COMPLETE | WB | 0 (reuse) | 0.60 | 0.25 | 0.15 |
| N-CLEAN | NB | 0 | 0 | 0 | 1 |
| N-COMPLETE | NB | 0 (reuse) | 0.60 | 0.25 | 0.15 |

Seed-0 complete cells reuse the promoted S0 checkpoints. No additional
complete-model seeds are in the active scope. New run directories are
immutable and unique. Test is never used for checkpoint selection.

## Exit rule

An item is addressed only when its raw evidence, aggregate output and
article-ready change agree. Unrun listening tests or target-device
measurements and the retained single-seed design remain explicit limitations,
not implied accomplishments.
