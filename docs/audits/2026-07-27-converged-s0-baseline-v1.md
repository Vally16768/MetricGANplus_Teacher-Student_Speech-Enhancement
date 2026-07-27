# Converged S0 baseline promotion audit

Evidence status: **historical, independently audited, superseded by v2 metrics**.

The three selected model files and their hashes remain valid. The aggregate
metrics below used padded batched inference and are retained only to document
the correction. Cite
`docs/audits/2026-07-27-converged-s0-baseline-v2.md` instead.

The canonical package is
`experiments/runs/20260727-converged-s0-baseline-v1`. It combines the pinned
official MetricGAN+ WB teacher with the WB and NB students selected after the
immutable max-50 continuation.

## Selection

| Cell | Selection protocol | Epoch-20 score | Selected score | Best/stop |
|---|---|---:|---:|---|
| S0-WB | `val_select` PESQ-WB | 2.596915 | 2.602952 | 34/42 |
| S0-NB | `val_select` PESQ-NB | 3.192184 | 3.216751 | 41/49 |

Both runs stopped through early stopping and neither is ceiling-limited.
PESQ-WB and PESQ-NB remain separate protocols.

## Public package boundary

Included:

- selected T0, S0-WB and S0-NB model packages;
- aggregate `val_rank`, `val_select` and test metrics;
- sanitized WB/NB histories and their plots;
- epoch-20 comparison, final plots, portable config and report;
- source, manifest, model and complete artifact SHA-256 inventories.

Excluded:

- VoiceBank+DEMAND audio and generated evaluation audio;
- teacher caches, replay buffers and noisy-score caches;
- training-state checkpoints;
- machine paths, usernames, host or mount information.

The failed first export, which was stopped because history files retained local
manifest/cache columns, is preserved only below the ignored local evidence
area. It was never staged or published. The corrected exporter removes those
two path fields while preserving every numeric history value.

## Validation

- source closure audit: 3/3 cells, 3/3 selected models, zero issues;
- canonical run contract: passed;
- promoted-package audit: zero issues;
- artifact inventory: 23 files before the manifest itself, approximately
  12.3 MiB total;
- privacy/exclusion scan: zero issues;
- selected model hashes match the audited local closure;
- no public file is 100 MiB or larger.

This package closes P3 only after the unit suite, canonical plan validation,
project guard, commit and push also pass.
