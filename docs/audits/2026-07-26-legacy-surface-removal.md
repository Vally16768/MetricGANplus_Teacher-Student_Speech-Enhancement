# Legacy public-surface removal — 2026-07-26

Status: removal authorized; recoverability verified before mutation.

## Recovery anchors

- public Git tree before cleanup: commit `5129bae`;
- local forensic copies: ignored `experiments/historical/`;
- dataset and local experiment runs: untouched.

The pre-cleanup commit is the immutable byte-level recovery source for every
tracked file removed below.

## Removed from the current public tree

- five superseded `code_and_documentation/README*.md` documents;
- DNS, combined-dataset, cross-domain, machine-runtime and obsolete scenario
  configurations under `code_and_documentation/configs/`;
- the complete 14-file Kingston/server runbook;
- five obsolete scenario/server orchestration scripts;
- DNS/FullSubNet/CMGAN orchestration from `repro.py`;
- FullSubNet, MP-SENet, CMGAN, aTENNuate, refiner and unrelated tiny-model
  builders from the canonical model registry.
- STM32 simulation/model modules; the ERB frontend operations still required
  by MetricGAN+ distillation were isolated in `sebench/erb.py`.

The canonical replacement is the repository-root `campaign.py` plus
`configs/voicebank_campaign.yaml`. It accepts VoiceBank+DEMAND only and exposes
WB teacher, WB student and NB student profiles.

## Non-loss statement

This cleanup removes files from the latest checkout, not from history. No
dataset, audio, local run, selected MetricGAN checkpoint or historical Git blob
was deleted. Restoration is possible from commit `5129bae`.
