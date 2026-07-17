# Data Pairing Policy (Teacher + Student)

## Scope
This document defines the current pairing policy used in this repository for cross-corpus training (DNS5 + VoiceBank+DEMAND).

## Current Default: `1:1`
- We use **one noisy sample per one clean sample** at manifest level (`noisy, clean` per row).
- This is the active default for our prepared combined manifests:
  - `train_combined.csv`
  - `val_rank_combined.csv`
  - `val_select_combined.csv`
  - `test_combined.csv`

## Exact Current Counts
- Combined manifest rows (pairs):
  - `train_combined.csv`: `463,771`
  - `val_rank_combined.csv`: `4,224`
  - `val_select_combined.csv`: `38,395`
  - `test_combined.csv`: `51,270`
- Combined total pairs: `557,660`

- By corpus inside combined manifests:
  - DNS5 pairs: `545,264`
  - VoiceBank+DEMAND pairs: `12,396`

- DNS5 source manifests (local):
  - `dns5-headset-16k/train.csv`: `504,463`
  - `dns5-headset-16k/val.csv`: `40,801`
  - Source total: `545,264`

This confirms that combined DNS5 coverage matches local DNS5 source manifests exactly (`545,264` pairs).

## Alignment with VoiceBank+DEMAND
- VoiceBank+DEMAND is used as a fixed paired corpus in this workflow.
- The practical setup is **1:1 paired supervision**, consistent with common VoiceBank+DEMAND usage.

## DNS5 in This Project
- DNS5 local manifests are treated as source-of-truth pairs (`noisy, clean`).
- We keep the resulting training/evaluation manifests in **1:1 row pairing** format for consistency with VoiceBank+DEMAND and reproducibility.

## Why We Keep `1:1` by Default
- Stable and reproducible training setup.
- Direct compatibility between DNS5 and VoiceBank+DEMAND in a single combined pipeline.
- Lower storage and runtime pressure versus larger `N:1` expansions.

## Note on `N:1` (Optional Future Extension)
- It is possible to expand to `N:1` (multiple noisy variants per clean) for training robustness.
- This is **not** the current default. Any switch to `N:1` must be explicitly configured and documented to preserve experiment comparability.
