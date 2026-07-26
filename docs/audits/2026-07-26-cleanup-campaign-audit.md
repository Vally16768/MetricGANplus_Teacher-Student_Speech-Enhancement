# Cleanup and campaign audit — 2026-07-26

Status: implementation/GPU smoke completed; authorized legacy removal applied;
commit/push validation in progress.

## Inventory

| Surface | Observed | Classification | Action |
|---|---:|---|---|
| canonical/control source | under 11 MB | keep | update and test |
| local historical imports | about 2.5 GB | private historical evidence | preserve locally; ignore in Git |
| local mutable runs | about 21 MB | private/regenerable | keep ignored |
| tracked machine runbooks | 14 files | private/machine-specific | removed; recoverable at `5129bae` |
| tracked DNS/combined configs | multiple | out of canonical scope | removed; recoverable at `5129bae` |
| alternative model families in monolith | present initially | out of canonical scope | removed from current model registry |
| reference checkpoints | present | historical/observed | retain until ancestry audit |

## Dataset observation

The external staging area contains VoiceBank audio, but the frozen CSV rows use
an older mount prefix. The source manifests have:

| Split | Rows | Source SHA-256 |
|---|---:|---|
| train_fit | 9,754 | `3a7a628cc567fd067db37dcde1155737203539f9790e7c37d8673230a2d78cd6` |
| val_rank | 128 | `091b8687f4b03d1ec20d39e3b72495be3a533ff5e648b8eb077a5bbf76ebe5fe` |
| val_select | 1,690 | `1358b9ec6a5ea622593b8f22beacdc1702a42041f06738774f1173d2ffedd9b9` |
| official test evidence | 824 | `33bc350117ce66de255f53edab080a9dce472927cdbb88f0f4029e18ec7fc495` |

The runtime created remapped copies under ignored `local/manifests/`. All
12,396 rows resolve, with no pair/clean duplicates and no cross-split overlap.
Neither source CSVs nor audio files changed.

## Implemented campaign evidence

- one portable controller with `validate`, `smoke-all`, `pilot-all`, `run-all`,
  `monitor-run` and `audit-run`;
- teacher WB baseline/metric pair from one anchor;
- separate WB/NB PESQ proxies and calibration outputs;
- one selected teacher with WB and NB cache targets;
- WB and NB student baseline/metric pairs;
- bandwidth-matched true metrics, curves, model hashes and report;
- GPU smoke `20260726-verification-smoke-wbnb-s0-a3` completed all six cells;
- 36/36 reported audio samples exist;
- independent package audit reports zero issues across six cells, six models
  and 36 samples;
- 28/28 unit/integration tests pass after adding pilot/monitor contracts.

This smoke is verification-only and cannot support an article claim.

## Cleanup policy

Cleanup is recoverable:

1. keep historical imports locally and hash-addressed;
2. prevent accidental inclusion with `.gitignore`;
3. build and validate the clean replacement flow;
4. only then remove tracked legacy files from the canonical surface;
5. retain their Git history and a concise audit index;
6. do not delete the local historical evidence without a later explicit,
   path-specific authorization.

## Remaining gates

- the cleanup must pass final smoke, project guard and Git review before commit;
- pilot must pass package/behavior audit before full training starts;
- held-out proxy calibration and multi-seed full experiments are not complete.
