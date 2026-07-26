# Data contract

## Logical sources

The canonical research line uses only paired noisy/clean speech from
VoiceBank+DEMAND with frozen manifests. DNS Challenge data and combined
VoiceBank/DNS experiments are historical, non-canonical evidence.

Never encode a personal filesystem location in committed files. Runtime
configuration must receive logical roots such as `METRICGAN_DATA_ROOT` through
an untracked local config.

## Read-only boundary

Dataset roots and source manifests are inputs. Allowed writes are limited to
repository-local ignored work areas or an explicitly configured output store.

The pipeline must reject:

- output/tracking/cache paths inside a dataset root;
- staging that copies or rewrites source audio without an explicit data-build
  task;
- a manifest rewrite performed in place;
- a split regenerated without a new manifest identity and audit.

## Split contract

| Split | Purpose | Selection allowed |
|---|---|---|
| `train_fit` | optimization | yes, training only |
| `val_rank` | frequent ranking | within run |
| `val_select` | final candidate selection | across declared candidates |
| `test` | final estimate | no tuning |

The same frozen pair identities and split membership feed all three profiles.
Audio is loaded read-only and resampled in memory:

- teacher WB and student WB: noisy and clean reference at 16 kHz;
- student NB: noisy and the matching clean reference at 8 kHz.

No resampled audio is written back to the dataset. Regenerable teacher caches
and run outputs belong under ignored/project artifact roots.

For each manifest record:

- row count;
- SHA-256;
- corpus/domain composition;
- speaker or identity rule;
- pair and clean-key duplicates;
- pair/clean overlap across splits;
- corrupt/missing audio count;
- generation code/commit.

The bound local copies were audited on 2026-07-26 without modifying source
manifests or audio:

| Split | Rows | Missing | Pair/clean duplicates |
|---|---:|---:|---:|
| `train_fit` | 9,754 | 0 | 0 |
| `val_rank` | 128 | 0 | 0 |
| `val_select` | 1,690 | 0 | 0 |
| `test` | 824 | 0 | 0 |

All pair and clean-identity overlaps between splits are zero. Exact
machine-local mapping remains in ignored `local/manifests/voicebank_v1/`; this
register retains only stable counts and protocol.

## Dataset change rule

This project currently forbids changing the dataset. A requested dataset change
must stop the workflow and require an explicit new decision, new manifest IDs,
new hashes, new split audit and invalidation analysis for every affected result.
