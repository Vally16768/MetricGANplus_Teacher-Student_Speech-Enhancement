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
- unit/integration suite: 30/30 passed on 2026-07-27 after cleanup, monitoring
  and WB/NB ERB extraction changes.
- stable post-cleanup VoiceBank-only six-cell GPU smoke: passed as
  `20260727-postcleanup-smoke-wbnb-s0-a5` on one NVIDIA GTX 1660 Ti.
- reported sample reconciliation: 36 unique paths, 36 files present.
- independent smoke package audit: six cells, six models, zero issues.
- smoke artifacts include true WB/NB metrics, proxy calibration, training
  curves, model hashes, aggregate CSV/JSON, plot and report.
- full VoiceBank-only pilot/training: blocked until a clean committed source
  snapshot exists; current smoke remains verification-only.
