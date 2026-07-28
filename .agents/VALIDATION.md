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
  sampling; CUDA smoke remains the next gate.
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
