# Experiment register

## Lifecycle

```text
planned -> smoke-passed -> running -> evaluated -> audited -> promoted
                                      \-> failed/invalid/superseded
```

Only `promoted` runs belong in the canonical article-facing set.

Working runs are created under ignored `local/runs/<run_id>/`. After validation,
a sanitized, self-contained research record is promoted to
`experiments/runs/<run_id>/`; private path mapping stays outside Git.

Promotion requires:

- clean committed source snapshot;
- unique run ID and immutable run directory;
- resolved config, exact command, seed and environment;
- dataset/manifest hashes and split audit;
- teacher and student checkpoint ancestry;
- complete logs and raw metrics;
- evaluation sample counts;
- model/checkpoint and integrity hashes;
- plots and concise report;
- status `valid` from independent validation.

Failed, invalid and superseded runs must not remain as ambiguous model folders.
Keep only the lesson, failure cause and recoverability reference in an audit.
Delete or externally archive bulk artifacts only after their targets and hashes
are verified and the user explicitly authorizes removal.

## Canonical run layout

```text
experiments/runs/<run_id>/
  provenance/
    provenance.json
    config_resolved.yaml
    command.txt
    environment.txt
  logs/
  metrics/
    summary.json
    per_sample.csv
  models/
  reports/
  status.json
```

## Registry

| Run/set | Status | Evidence | Canonical |
|---|---|---|---|
| public reference checkpoints | observed | current repository | no |
| imported Kingston results | pending cleanup/audit | hash-verified copies | no |
| imported legacy worktree | historical dirty snapshot | hash-verified copies | no |
| `smoke_local_setup` | verification-only | `prepare_data` passed | no |
| `20260726-verification-smoke-wbnb-s0-a1` | failed smoke | CUDA device lacked explicit index; no dataset writes | no |
| `20260726-verification-smoke-wbnb-s0-a2` | superseded smoke | six cells passed, but reported audio samples were not retained | no |
| `20260726-verification-smoke-wbnb-s0-a3` | smoke-passed/audited | six cells, WB/NB proxies, dual cache, 36/36 samples, six model hashes, zero audit issues | no |
| `20260726-postcleanup-smoke-wbnb-s0-a4` | superseded smoke | six cells completed, but ERB extraction changed the working source while the process was active; proxy WB held-out correlation was negative on six smoke records | no |
| `20260727-postcleanup-smoke-wbnb-s0-a5` | smoke-passed/audited | stable post-cleanup source; six cells, six models, 36/36 reported samples, matched WB/NB protocols, zero audit issues | no |

There is currently no promoted end-to-end run from the current repository
snapshot.

The `a3` smoke proves wiring only. It used 8 training pairs and 4 pairs per
evaluation split for one epoch. Its metric values and proxy calibration are
not scientific evidence and must not enter the article.

## Required reporting

Report PESQ, STOI, SI-SDR, delta-SNR and support counts for each split/domain.
Every metric row must include `bandwidth`, `sample_rate`, `pesq_mode` and
`reference_bandwidth`. Never average WB-PESQ and NB-PESQ into one number.
For declared multi-seed experiments report mean, standard deviation and 95%
confidence interval. Add model parameters, serialized size, latency protocol
and measured latency for deployment claims.

Graphs must include:

- train/validation loss;
- selection metric by epoch;
- teacher vs student vs QAT comparison;
- per-domain final metrics;
- latency/quality trade-off when deployment is claimed.

## Canonical ablation matrix

The first campaign is a controlled six-cell comparison. Architecture, split,
seed set and optimizer schedule stay fixed within each paired comparison.

| Cell | Model | Band | Loss | Comparison |
|---|---|---|---|---|
| T-WB-BASE | WB teacher | WB | `T0` | teacher baseline |
| T-WB-METRIC | WB teacher | WB | `T0_PESQ` | metric-discriminator effect |
| S-WB-BASE | WB student | WB | `D1` | WB distillation baseline |
| S-WB-METRIC | WB student | WB | `D1_PESQ` | WB student metric effect |
| S-NB-BASE | NB student | NB | `D1` | NB distillation baseline |
| S-NB-METRIC | NB student | NB | `D1_PESQ` | NB student metric effect |

Primary effect sizes are paired deltas within profile:
`METRIC - BASE` for PESQ, STOI, SI-SDR and delta-SNR. Teacher improvement is
established first. Student comparisons use the promoted WB teacher and their
own bandwidth-calibrated proxy.

Required metric-proxy evidence per bandwidth:

- label-generation manifest and PESQ protocol;
- train/validation split by utterance identity;
- proxy MSE/MAE and rank correlation on held-out candidates;
- calibration plot and score-range coverage;
- frozen proxy checkpoint hash;
- generator ablation showing true-metric change, not only predicted change.

The TTS extension is a separate future campaign. It may reuse the objective
adapter but requires a selected TTS generator, its own outputs, a recalibrated
proxy and separate dataset/provenance. It cannot be promoted from the
VoiceBank enhancement results.
