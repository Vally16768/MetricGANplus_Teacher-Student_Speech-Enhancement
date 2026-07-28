# Validation matrix

## Gates by change type

| Change | Required validation |
|---|---|
| Documentation only | documentation index + project guard |
| Config/path | unit tests + unsafe-path test + `prepare_data` smoke |
| Model/loss | unit tests + forward/backward smoke + checkpoint compatibility |
| Data/split logic | manifest hashes + overlap/duplicate/support audit |
| Training loop | unit tests + real entry-point short train + resume test |
| Evaluation/metrics | fixture metric test + sample-count reconciliation |
| Bandwidth/profile | explicit WB/NB contract + reference/mode metadata test |
| Metric discriminator | exact layer contract + normalized labels + current/history/current update + local replay/no-input-copy test + frozen-during-G check + current-output calibration + true-metric ablation |
| Official checkpoint | pinned revision + SHA-256 + exact tensor mapping + offline package round-trip + true PESQ diagnostic |
| Teacher cache | outside dataset + no input duplication + dtype/error bound + resume + manifest fallback |
| Architecture | all relevant tests + architecture register/hash update |
| Experiment promotion | run contract + independent metric recomputation |

## Commands

```bash
PYTHON=/path/to/shared-venv/bin/python

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/validate_research_plan.py
"$PYTHON" .agents/skills/manage-metricgan-research/scripts/project_guard.py --repo .
"$PYTHON" .agents/skills/manage-metricgan-research/scripts/run_contract.py \
  validate --repo . --run-dir experiments/runs/<run_id> --stage canonical
```

Runtime-sensitive changes also require the real entry point:

```bash
"$PYTHON" campaign.py validate
"$PYTHON" campaign.py smoke-all --run-id <verification-id> --allow-dirty-smoke
"$PYTHON" campaign.py smoke-baseline --run-id <verification-id> --allow-dirty-smoke
"$PYTHON" campaign.py run-baseline --run-id <baseline-id>
"$PYTHON" campaign.py monitor-run --run-dir local/runs/<run-id>
"$PYTHON" campaign.py audit-run --run-dir local/runs/<run-id>
```

Training changes require a bounded GPU smoke using the same entry point and a
new output directory. Never launch a matrix from a dirty worktree.

Training preflight additionally requires:

```text
active sys.prefix == METRICGAN_SHARED_VENV or repository-sibling shared-venv
requested device starts with cuda
torch.cuda.is_available() == true
dataset identity == VoiceBank+DEMAND
teacher profile == WB/16 kHz
metric/reference profile matches the run profile
```

## Current baseline

- converged S0 closure `20260727-converged-s0-baseline-a1`: 3/3 cells and
  models, 54 local report samples, zero independent-audit issues;
- CUDA resume-equivalence A4: injected post-evaluation interruption; LR,
  patience, best state, selected-model hash and history equal to the control;
- portable S0 package `20260727-converged-s0-baseline-v2`: exact v1 model
  hashes re-evaluated per utterance at true length; canonical run contract,
  complete artifact-manifest audit and privacy scan passed; no dataset/audio/
  cache/replay/training-state artifact;
- unit/integration suite: 56/56 passed after portable promotion and manifest
  tamper-detection coverage;
- canonical research plan and project guard: passed with zero issues.
- strict teacher-calibration A2: audited with zero package issues and zero
  generator updates; its 100-record held-out gate failed and remains negative
  evidence.
- two-refresh calibration smoke A1: 2/2 refreshes, zero generator updates,
  epoch rows contain no redundant evaluation, independent audit zero issues.
- unit/integration suite: 62/62 passed after the predeclared calibration-retry
  and frozen-evaluation reuse change.
- final strict calibration A3: 2/2 refreshes on 100 update plus 100 disjoint
  held-out outputs; final gate failed at normalized MAE 0.2133, Pearson 0.5545
  and Spearman 0.5435; zero generator updates and zero audit issues.
- T6 production search selected exact T5 at scale `1.0`; PESQ `+0.005075`,
  STOI `-0.000782`, SI-SDR `-0.245701` dB; gate failed and test was unread.
- T7 implementation: 90/90 unit/integration tests passed, including disabled
  parity, exact confidence formula and portable checkpoint round-trip;
  campaign split validation and project guard passed.
- T7 clean CUDA smoke `20260728-t7-confidence-smoke-wb-s3003-a1` completed
  fit/cal/rank/select, T0 fallback and checkpoint round-trip; verification-only,
  two-file support and no test read.
- T7 production `20260728-t7-confidence-wb-s3003-a1` completed with checkpoint
  round-trip and no test read; `+0.004931` PESQ missed the teacher gate.
- T8 implementation: 92/92 unit/integration tests passed, including ridge
  recovery, exact T0/T7 routing and portable checkpoint round-trip; campaign
  split validation and project guard passed.
- T8 clean CUDA smoke `20260728-t8-router-smoke-wb-s3003-a1` completed 10/10
  fit/cal labels, ridge, threshold selection, round-trip and the intended
  pre-validation stop; verification-only and no validation/test read.
- T8 production `20260728-t8-router-wb-s3003-a1` learned `+0.009197` PESQ
  with safe calibration guardrails, but its exact `+0.014197` oracle ceiling
  missed the frozen `+0.015` gate; it stopped before validation and test.
- T9 implementation: 94/94 unit/integration tests pass, including exact
  multi-action selection, checkpoint round-trip and fresh calibration
  partition/clean disjointness; campaign split validation and project guard
  pass.
- T9 clean CUDA smoke `20260728-t9-router-smoke-wb-s3003-a1` completed 10/10
  fit/cal labels, four ridge models, multi-action selection and checkpoint
  round-trip. Its small-support guardrail failure caused the intended stop
  before validation; it is verification-only and test remained unread.
- T9 production `20260728-t9-router-wb-s3003-a1` completed 256/128 labels,
  four ridge models and checkpoint round-trip. Oracle gain was `+0.031116`;
  no learned threshold met both auxiliary guards, so validation/test remained
  unread.
- T10 implementation: 95/95 unit/integration tests pass, including fresh
  T3-audit support separation and inherited exact router/checkpoint behavior.
  Campaign split validation and project guard pass; clean CUDA smoke remains
  pending.
- T10 clean CUDA smoke `20260728-t10-router-smoke-wb-s3003-a1` completed
  fresh support, conservative-margin selection, rank/select flow and
  checkpoint round-trip. It is verification-only; test remained unread.
- T10 production `20260728-t10-router-wb-s3003-a1` passed prevalidation,
  rank and both final auxiliary guards; `val_select` PESQ gain `+0.008015`
  missed the unchanged gate, and test remained unread. T11 validation is
  pending.
- T11 implementation: 96/96 unit/integration tests pass, including immutable
  strength-penalty folding and inherited router/checkpoint behavior. Campaign
  validation and project guard pass; clean CUDA smoke remains pending.
- T11 clean CUDA smoke `20260728-t11-router-smoke-wb-s3003-a1` completed fresh
  support, all 25 penalty/margin policies and checkpoint round-trip. The
  small-support PESQ gate caused the intended stop before validation; test was
  unread.
- T11 production `20260728-t11-router-wb-s3003-a1` selected penalty `0.04`
  and margin `0.005`; rank gained `+0.010068` PESQ with safe auxiliaries, but
  `val_select` gained only `+0.008349`. The gate failed and test was unread.
- T12 implementation adds one exact `val_rank` four-action pass, a predeclared
  72-policy ranking surface, deterministic selection, checkpoint round-trip
  and one conditional `val_select` evaluation. Full suite `97/97` and
  VoiceBank campaign split validation and project guard pass.
- T12 clean CUDA smoke `20260728-t12-router-smoke-wb-s3003-a1` completed ten
  rank examples, all 72 policies, guardrail-aware selection and checkpoint
  round-trip. It was verification-only and read neither `val_select` nor test.
- T12 production `20260728-t12-router-wb-s3003-a1` selected penalty `0.015`
  and margin `0.015`; rank gained `+0.011176`, but `val_select` gained only
  `+0.008425` PESQ. Auxiliary guards passed and test remained unread.
- T13 implementation full suite `98/98` and VoiceBank campaign split
  validation pass; architecture hashes and project guard pass.
- T13 clean CUDA smoke `20260728-t13-router-smoke-wb-s3003-a1` completed
  train/rank labeling, twelve metric-delta ridges, 336 policies and checkpoint
  round-trip. Small support caused the intended prevalidation stop; select and
  test were unread.
- final evidence reconciliation: canonical v2 contract/audit, local A3 audit,
  sanitized negative JSON source hashes and privacy scan all passed.
- T2 parity audit identified that the historical T1 discriminator frontend
  used sqrt-magnitude, reflect padding and frequency-first layout instead of
  SpeechBrain's magnitude, constant padding and time-first layout. T1 remains
  valid negative evidence for that implementation and cannot seed D2.
- corrected T2 discriminator parity: SpeechBrain v1.1.0 revision `36c180c`;
  fixed-tensor frontend and imported-state output parity passed; exact
  clean/enhanced/noisy plus historical update trace passed; true-length batch
  invariance passed; complete suite `65/65`, canonical plan/campaign
  validation and project guard passed.
- direct shared-venv CUDA smoke on NVIDIA GTX 1660 Ti produced finite D
  parameter gradients and, after freezing D, a finite candidate waveform
  gradient with no D gradients.
- fixed D2 support `20260728-t2-d2-support-s0-a3` independently reconciles
  1000 train, 200 calibration and 200 untouched audit records; 1400 unique
  pair/utterance tokens, zero missing or non-FP16 T0 targets, zero non-finite
  fields, no copied inputs and unchanged source hashes. Support SHA-256:
  `545ac1bfa2ad4075d89ecdcf89f1dd138e1524df93a4ac9065076aa75238e3e4`.
- the source manifests do not retain VoiceBank speaker/noise identities;
  exact speaker-disjointness is not claimed for D2 support. Dataset-level
  frozen split overlap remains zero.
- D2 interrupted/resumed unit control matches the uninterrupted model,
  scheduler, history, best epoch and best score exactly; complete suite
  `67/67`, config/plan/campaign validation and project guard pass.
- clean CUDA smoke `20260728-t2-d2-official-smoke-s0-a1` observed batch-1
  current/history/current passes, true normalized labels, selected checkpoint,
  complete state and calibration/directional plots. Its two-record relaxed
  gate is verification-only and not scientific D2 evidence.
- strict D2-OFFICIAL `20260728-t2-d2-official-s0-a1` stopped at epoch 6
  (best epoch 1) and failed safely: audit nMAE `0.2895`, Pearson `0.7626`,
  Spearman `0.7768`; local sign agreement `0.5291`, delta Spearman `-0.4929`;
  zero teacher updates.
- D2-RANGE train-only support/fitting implementation: `68/68` tests, including
  fixed-audit isolation, FP16-only derived candidates and balanced PESQ-bin
  sampling; production CUDA smoke passed.
- D2-RANGE support `20260728-t2-d2-range-support-s0-a1`: 7,000 candidates,
  1,000 train parents, seven declared types, 200 fixed audit identities, zero
  missing/non-FP16/non-finite artifacts; support SHA-256 `3d563216...`.
- strict D2-RANGE `20260728-t2-d2-range-s0-a1`: best epoch 6, early stop 11;
  audit nMAE `0.3287`, Pearson `0.7283`, Spearman `0.7599`; local sign
  `0.3266`, delta Spearman `-0.6221`; failed safely with zero G updates.
- sanitized D2-OFFICIAL/D2-RANGE negative packages: 12/12 listed files each,
  hashes and sizes reconcile, no private path in text/checkpoints, no file
  above 100 MiB; final `68/68` tests, plan/config validation and project guard
  pass.
- T3 plan/control-plane update: direct PMSQE plus MR-STFT/SI-SDR/trust-region
  branch, conditional pairwise critic and strict downstream teacher/student
  gates are indexed; project skill quick validation, canonical plan validator,
  documentation diff check and project guard pass with zero issues.
- T3.1 direct-loss implementation: pinned `torch-pesq==0.1.2` and its MIT
  source/core hashes; isolated true-length E1/E2 objective; short/silent/
  padding/amplitude and CUDA AMP coverage; full suite `77/77` passed.
  A real VoiceBank 32,000-sample CUDA smoke produced finite PMSQE `2.3363`,
  waveform-gradient norm `67.7028` and finite gradients for all 21 official
  teacher tensors (aggregate norm `1.1645`). No optimizer step was taken.
- T3 support/control implementation: root-entry-point commands freeze and
  independently audit pair/clean-disjoint train identities, exclude supplied
  T2 supports, and calibrate weights only on 16 frozen train directions under
  CUDA. Full suite `78/78` and campaign validation pass; scientific support
  generation remains pending on a clean commit.
- First T3 weight-calibration attempt stopped before output because the helper's
  internal torch-pesq buffers remained on CPU for a CUDA candidate. Zero model
  or optimizer updates occurred. The helper now moves every component module
  to the candidate device and the focused CUDA calibration regression passes.
- T3 fixed support `20260728-t3-direction-support-s0-a1`: 1,000 train, 200
  calibration and 200 untouched audit identities; pair and clean-utterance
  disjoint from both supplied T2 supports; identity hash `04022b77...`;
  independent audit zero issues. Train-only 16-row CUDA calibration froze
  anchor `4.30122085`, PMSQE `0.00186623`, SI-SDR `0.10`; weights hash
  `e9edaae1...`; validation/test rows used `0/0`.
- T3 candidate/direction implementation predeclares four bounded mask-logit
  deltas, zero-delta/cache parity, FP16-only output storage and the 200-pair
  untouched audit gate. Synthetic perfect-direction fixture passes all gate
  components; focused suite `10/10`, full suite `79/79`, campaign validation
  and project guard pass.
- The scientific T3 direction gate additionally binds finite, non-vanishing
  waveform-component gradients and a surrogate gradient reaching every
  trainable official-teacher tensor. The audit cannot pass on correlation
  alone.
- Completed T3 direct-loss direction evidence: 5,600 fixed FP16 mask-logit
  candidates; zero-delta/cache MAE maximum `1.024e-05`; untouched audit has
  771 eligible pairs, sign agreement `0.9222`, delta Spearman `0.8982` and
  minimum SNR-quartile agreement `0.8454`. All direct-loss and artifact gates
  passed; E2 is eligible for the matched teacher pilot.
- T3 matched trainer: focused `12/12` T3 tests pass, including bit-exact
  post-evaluation resume for model, Adam, plateau scheduler and RNG.
  Full repository suite `81/81` passes. `20260728-t3-e1-e2-smoke-a1`
  completed E0/E1/E2 on CUDA from the exact
  official/cache/support hashes; both branches accepted one update and E2
  passed its current-output direction recheck. Its two-file PESQ delta
  (`-0.000094`) is verification-only and is not a teacher claim.
- First full T3 attempt `20260728-t3-e1-e2-full-s3003-a1` is invalid
  infrastructure evidence: deterministic CUDA was enabled after E0 without
  first setting the CuBLAS workspace contract. E1 recorded three identical
  pre-update failures and zero optimizer steps; the run was stopped before E2.
  The corrected entry point sets determinism before E0 and propagates
  unexpected runtime failures instead of classifying them as scientific
  rollback decisions.
- Corrected contract-adoption smoke
  `20260728-t3-e1-e2-contract-smoke-a2` passed the complete E0/E1/E2 CUDA
  path on the clean fix commit: both deterministic updates were accepted,
  E2 current-direction passed, and the run stayed verification-only.
- Full corrected T3 run `20260728-t3-e1-e2-full-s3003-a2` completed from the
  exact official T0 and frozen support. E1 and E2 each rejected and rolled
  back proposals at learning rates `1e-6`, `5e-7` and `2.5e-7`; both selected
  the exact T0 hash and had zero `val_select` gain. This is valid negative
  evidence, not a promoted teacher.
- T4-A focused tests verify that a folded `linear2.bias` produces the exact
  bounded mask-logit variant and rejects shifts outside `+/-0.10`.
- T4-A pre-run validation passes `83/83` repository tests, the canonical
  research-plan validator, the real `campaign.py validate` entry point,
  architecture-source hash reconciliation, project guard and diff check.
- Contracted T4-A run `20260728-t4-logit-bias-wb-s3003-a1` completed all ten
  `val_rank` candidates and one selected `val_select` evaluation. Its artifact
  hashes reconcile, `test_read=false`, both guardrails pass and the
  `+0.002034` PESQ gain correctly fails the unchanged `+0.01` gate.
- T4-B implementation validation passes `85/85` tests. Coverage includes the
  exact bounded state interpolation and the declared PMSQE-primary constrained
  loss composition. Research-plan/campaign validation, architecture hashes,
  project guard and diff check pass before the clean CUDA smoke.
- Clean contracted CUDA smoke
  `20260728-t4b-microstep-smoke-wb-s3003-a1` completed an exact-T0 one-step
  trajectory, alpha `1/.5` interpolation, ordinary checkpoint round trip and
  two-file WB rank/select evaluation. Metrics and gradients remained finite;
  `production_support=false`, `verification_only=true` and `test_read=false`.
- Full T4-B run `20260728-t4b-microstep-wb-s3003-a1` completed every declared
  horizon/interpolation with production support. The 256-step full proposal
  became unsafe; the selected H1/alpha-0.125 candidate had `val_select` PESQ
  delta `-0.00000119` and correctly failed the unchanged gate. Test, cache and
  students remained unread/unbuilt.
- T5 pre-run implementation validation passes `87/87` tests, including
  uniform-curve/scalar equivalence, coefficient bounds and pair/clean-disjoint
  train-only fit/calibration manifests. Research-plan/campaign validation,
  architecture hashes, project guard and diff check pass.
- Clean contracted T5 smoke `20260728-t5-frequency-smoke-wb-s3003-a1`
  completed fit-only coordinate selection, disjoint calibration, rank
  rejection and T0 fallback on CUDA. It is explicitly two-file,
  `production_support=false`, `verification_only=true` and `test_read=false`.
- Full T5 run `20260728-t5-frequency-wb-s3003-a1` completed 48 fit decisions,
  three disjoint-calibration sweeps, five `val_rank` candidates and one
  `val_select` evaluation. Selected sweep 3 gained `+0.005075` PESQ with both
  guards passing, but correctly failed the unchanged `+0.01` gate; test was
  unread.
- T6 affine-logit implementation passes `88/88` tests, including exact
  final-layer `scale * logits + curve` folding. Campaign validation,
  architecture hashes, project guard and diff check pass.
- `campaign.py --help`: passed in the shared project environment.
- safe I/O config: passed.
- output-under-dataset guard: blocked as expected.
- `prepare_data` on current runtime manifests: passed.
- source manifest hashes before/after smoke: unchanged.
- canonical WB/NB research-plan validation: passed.
- WB/NB family aliases and mismatch rejection: passed.
- metric-objective teacher/student backward smoke: passed.
- local shared environment prefix and CUDA resolution: passed on 2026-07-26.
- safe checkpoint/proxy restricted-load round trips: passed.
- unit/integration suite: 39/39 passed on 2026-07-27 after the official-teacher,
  two-stage campaign and local FP16 cache changes.
- stable post-cleanup VoiceBank-only six-cell GPU smoke: passed as
  `20260727-postcleanup-smoke-wbnb-s0-a5` on one NVIDIA GTX 1660 Ti.
- reported sample reconciliation: 36 unique paths, 36 files present.
- independent smoke package audit: six cells, six models, zero issues.
- smoke artifacts include true WB/NB metrics, proxy calibration, training
  curves, model hashes, aggregate CSV/JSON, plot and report.
- clean-snapshot pilot `20260727-pilot-wbnb-s0-a1`: passed on commit `76729f3`;
  six cells, six models, 72/72 reported samples and zero audit issues.
- pilot manifest hashes were identical before/after execution; split audit
  retained zero pair/clean overlaps.
- pilot WB/NB metric/reference contracts passed and held-out proxy Pearson
  correlations were 0.9695/0.9551.
- first full attempt `20260727-full-wbnb-s0-a1` was deliberately stopped and is
  non-promotable; artifacts and hashes are preserved.
- causal-max structural audit passed: WB `604386` parameters, NB `514018`,
  GRU hidden size `160`, three layers, linear size `224`, fixed 16 ms
  lookahead.
- official teacher structure passed: exact 512/256/512 Hamming/log-magnitude
  frontend, 1,895,514 parameters, 21/21 mapped generator tensors and zero
  skipped tensors.
- official checkpoint package round-trip passed without accessing the remote
  model/cache loader.
- pinned official checkpoint loaded on CUDA and produced PESQ-WB 3.340675,
  STOI 0.933437 and SI-SDR 9.192374 on the frozen four-row smoke test support.
- local teacher-cache tests passed: FP16 teacher waveform/mask payloads,
  no noisy/clean copies, float32 loader output, maximum waveform quantization
  error below 0.0005 and successful resume validation.
- 39/39 unit/integration tests and the research-plan validator passed after the
  architecture and campaign change.
- failed-gate fallback test passed: verification runs record the rejected T1
  candidate but feed `T0-WB-OFFICIAL` downstream.
- direct two-second WB and NB forward/backward passes completed on the shared
  venv's NVIDIA GTX 1660 Ti with finite gradients.
- clean-snapshot two-stage smoke A3 passed on commit `8d36d62`: seven cells,
  seven hashed models, 42/42 sample files, zero audit issues and unchanged
  manifest hashes.
- the smoke correctly rejected both one-epoch T1 branches, retained
  `T0-WB-OFFICIAL` downstream and remained non-promotable.
- S0/S1 reused one physical 2.9 MiB cache when the teacher checkpoint was
  unchanged; the cache recorded both stage labels and no dataset audio copies.
- clean-snapshot two-stage pilot A1 passed execution/audit on commit `0756a68`:
  seven cells/models, 84/84 sample files, zero audit issues, unchanged
  manifests and zero split overlap.
- pilot T0 test PESQ-WB/STOI were 3.2626/0.9266 on 64 pairs; the WB proxy
  calibrated to Pearson 0.9539 and Spearman 0.9325 on 96 held-out candidates.
- both T1 branches reduced true `val_select` PESQ despite optimization; the
  gate retained T0 and the full campaign remains blocked.
- bounded MetricGAN target-score and T0 trust-anchor unit tests passed; the
  complete suite is 41/41 after the corrective loss change.
- bounded-teacher clean GPU smoke A1 passed on commit `27838d9`: seven cells,
  seven models, 42/42 samples and zero independent-audit issues.
- bounded T1 control/metric epoch deltas were only -0.0004/-0.0007 PESQ-WB,
  eliminating the earlier ~0.4 PESQ collapse; neither passed the gain gate.
- bounded-teacher clean pilot A1 passed execution/audit on commit `33ef895`:
  seven cells/models, 84/84 samples and zero issues.
- pilot control PESQ moved 2.8238→2.8093→2.7964; bounded metric PESQ moved
  2.8238→2.8197→2.8131. Both retained epoch-0 T0 and blocked full.
- at that historical stage, the next gate was an alternating
  current/noisy/historical discriminator refresh smoke and pilot; both were
  subsequently executed below.
- alternating implementation unit tests cover PESQ normalization, SpeechBrain
  spectral-normalized layer/checkpoint contract, three-pass refresh, FP16
  generated-output replay, no noisy/clean copies and D freezing after refresh.
- 45/45 unit/integration tests, the canonical research-plan validator, the real
  `campaign.py validate` entry point and the project guard passed before the
  later clean GPU evidence recorded below.
- first clean alternating smoke A1 completed and audited 7/7 cells, 42/42
  samples and zero issues, but exposed a warm-start-only clean-label mismatch:
  library PESQ could exceed 4.5 while official D_clean is exactly 1. The run is
  preserved as structural evidence and superseded before pilot.
- corrected alternating smoke A2 on clean commit `f5003ef` passed 7/7 cells,
  42/42 samples and zero independent-audit issues. The clean=1 target, all
  three D passes, D freeze for G, resumable D checkpoint and local
  generated-only FP16 replay were observed.
- alternating pilot A1 on clean commit `9ad2b85` completed and independently
  audited 7/7 cells/models, 84/84 samples and zero issues.
- the pilot kept WB/NB references and PESQ modes aligned, and its local 6.9 MiB
  replay contained 64 generated FP16 outputs with no noisy/clean input copies.
- T1 gained only +0.00221 PESQ-WB on `val_select`, below the +0.01 gate, while
  current-output D MAE degraded from 1.5002 to 1.7555; T0 fallback passed and
  full training remains blocked.
- baseline-only audit fixtures reconcile exactly three expected cells and do
  not require a T1 promotion gate; a full baseline still requires a clean
  committed snapshot and its own independent package audit.
- baseline-only CUDA smoke A1 completed the exact T0/S0-WB/S0-NB scope,
  produced 3/3 hashed models and 18/18 sample files, and passed the independent
  audit with zero issues.
- the smoke package contains no proxy/T1/S1 cells; its WB/NB metric/reference
  metadata match and its reused FP16 cache contains zero noisy/clean input
  files.
- the 50-epoch student policy and immutable two-cell continuation package pass
  48/48 unit/integration tests; coverage includes the exact plateau/early-stop
  contract, dynamic continuation audit, source state/model hashes and the
  historical baseline audit contracts.
- canonical configuration validation, research-plan validation and the project
  guard pass after the student schedule/resume change.
- clean-snapshot real-entry-point CUDA smoke
  `20260727-student50-policy-smoke-s0-a1` passed on commit `330e501`: exact
  T0/S0-WB/S0-NB scope, 3/3 cells/models, 18/18 samples, matched WB/NB
  protocols and zero independent-audit issues.
- full baseline `20260727-official-baseline-full-s0-a1` passed its independent
  package audit with 3/3 cells/models, 54/54 reported samples and zero issues.
- immutable continuation `20260727-official-students-cont50-s0-a1` completed
  and audited with 2/2 cells/models, 36 report samples and zero issues. WB
  selected epoch 34 and early-stopped at 42; NB selected epoch 41 and
  early-stopped at 49. Neither is ceiling-limited.
- the updated project skill and iterative execution board pass the skill
  validator, 48/48 unit/integration tests and the project guard with zero
  issues; this was the pre-closure control-plane checkpoint.
