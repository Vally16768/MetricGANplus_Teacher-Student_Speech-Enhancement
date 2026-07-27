# Final results — official T0 and converged S0 students

Status: **canonical corrected baseline plus reproduced negative T1 diagnostic**.

This document is the article-facing results index for the completed P1–P6
sequence. Detailed architecture, data and experimental contracts remain in
`.agents/`; raw public baseline artifacts are in
`experiments/runs/20260727-converged-s0-baseline-v2`.

## Research outcome

The official MetricGAN+ WB checkpoint successfully supervised converged WB and
NB causal students. The proposed T1 phase did not produce an improved teacher:
its metric discriminator failed current-output calibration before any
generator update. Therefore S1 students were correctly not trained.

## Canonical S0 results

| Cell | Profile | Best/stop | `val_select` PESQ | `val_select` STOI | `val_select` SI-SDR | Test PESQ | Test STOI | Test SI-SDR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T0-WB-OFFICIAL | WB / PESQ-WB | official | 2.785350 | 0.868438 | 5.414771 | 3.130948 | 0.931896 | 8.579135 |
| S0-WB | WB / PESQ-WB | 34 / 42 | 2.603164 | 0.854977 | 5.534107 | 3.051914 | 0.929624 | 9.049854 |
| S0-NB | NB / PESQ-NB | 41 / 49 | 3.216944 | 0.853526 | 5.598627 | 3.615133 | 0.929400 | 9.070931 |

Support is 1,690 paired utterances for `val_select` and 824 for test.
PESQ-WB and PESQ-NB are different protocols and are not compared or pooled.
All values use per-utterance true-length inference. The study has one training
seed, so it does not establish across-seed confidence intervals.

Relative to the original 20-epoch ceiling, S0-WB improved `val_select`
PESQ-WB by 0.006248 and S0-NB improved PESQ-NB by 0.024760. Both runs stopped
through early stopping before epoch 50 and are not ceiling-limited.

## T1 stop gate

The final calibration refresh used 100 update outputs and 100 disjoint
held-out current E0 outputs. It produced:

| Metric | Required | Observed | Result |
|---|---:|---:|---|
| normalized PESQ MAE | ≤ 0.06 | 0.2133 | fail |
| Pearson | ≥ 0.80 | 0.5545 | fail |
| Spearman | ≥ 0.80 | 0.5435 | fail |
| held-out count | ≥ 100 | 100 | pass |
| non-constant predictions | std ≥ 0.02 | 1.0063 | pass |
| calibrated score range | tolerance ≤ 0.30 | 0.7588–5.2196 vs 1.5041–3.9593 | fail |

There is no T1−T0 or S1−S0 effect size because T1 was never accepted and S1
was never run. Reporting those comparisons as zero would be incorrect.

## Claim-to-artifact map

| Claim | Evidence |
|---|---|
| S0 models converged under max-50 policy | v2 histories, selection metadata and convergence plot |
| WB/NB metrics use matched references and PESQ modes | v2 canonical metrics CSV and campaign summary |
| Public model hashes equal audited local selections | v2 model inventory and import manifest |
| Padding no longer changes recurrent evaluation | true-length regression test and v2 corrective audit |
| T1 discriminator was unsafe for G updates | A3 two-refresh calibration metrics and negative audit |
| No downstream work occurred after gate failure | zero G updates; no E1/E2/C1/S1 artifacts; execution board |

## Article assets and limitations

- baseline table source:
  `experiments/runs/20260727-converged-s0-baseline-v2/metrics/canonical_metrics.csv`;
- convergence figure:
  `experiments/runs/20260727-converged-s0-baseline-v2/reports/convergence_comparison.png`;
- test PESQ figure:
  `experiments/runs/20260727-converged-s0-baseline-v2/reports/test_pesq_by_cell.png`;
- baseline correction audit:
  `docs/audits/2026-07-27-converged-s0-baseline-v2.md`;
- T1 negative audit:
  `docs/audits/2026-07-27-teacher-calibration-t1-negative.md`.

The principal limitations are one seed, no listener study, no uncertainty
across retrainings and a failed T1 calibration stage. The separate TTS
metric-critic hypothesis remains untested and must not be inferred from these
speech-enhancement results.
