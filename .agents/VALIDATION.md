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
| Metric discriminator | proxy calibration + frozen weights + generator gradient smoke + true-metric ablation |
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
- the next required gate is a tested bounded/refreshable teacher metric
  objective followed by a new clean smoke and monitored pilot.
