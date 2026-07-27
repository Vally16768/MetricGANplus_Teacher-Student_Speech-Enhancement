# Converged official-teacher S0 baseline closure

Evidence status: **reproduced and audited candidate**.

The official MetricGAN+ WB teacher is unchanged. The two students were continued from immutable epoch-20 optimizer states under the declared max-50, plateau-LR and early-stopping policy.

## Epoch-20 versus converged selection

| Cell | Protocol | Epoch-20 | Converged | Delta | Stop |
|---|---|---:|---:|---:|---|
| S0-WB | PESQ-WB | 2.596915 | 2.602952 | +0.006037 | best 34, early_stopping at 42 |
| S0-NB | PESQ-NB | 3.192184 | 3.216751 | +0.024567 | best 41, early_stopping at 49 |

## Final profile-matched metrics

| Cell | Split | PESQ | STOI | SI-SDR | Delta-SNR | Support |
|---|---|---:|---:|---:|---:|---:|
| T0-WB-OFFICIAL | val_rank | 2.717322 | 0.862567 | 5.513523 | -0.014150 | 128 |
| T0-WB-OFFICIAL | val_select | 2.698865 | 0.861746 | 5.403473 | -0.368739 | 1690 |
| T0-WB-OFFICIAL | test | 3.122664 | 0.931190 | 8.482982 | -0.703872 | 824 |
| S0-WB | val_rank | 2.598257 | 0.856924 | 5.653440 | 0.238841 | 128 |
| S0-WB | val_select | 2.602952 | 0.854980 | 5.534464 | -0.153600 | 1690 |
| S0-WB | test | 3.051937 | 0.929625 | 9.050483 | -0.156349 | 824 |
| S0-NB | val_rank | 3.198166 | 0.855797 | 5.756929 | 0.316484 | 128 |
| S0-NB | val_select | 3.216751 | 0.853534 | 5.600131 | -0.114468 | 1690 |
| S0-NB | test | 3.615061 | 0.929402 | 9.072483 | -0.141557 | 824 |

WB and NB PESQ use different bandwidth protocols and are never pooled or directly ranked against one another.

Both students stopped through early stopping and neither selected the epoch-50 ceiling. The result is one-seed evidence; uncertainty across seeds is not established.

Source packages:
- baseline: `20260727-official-baseline-full-s0-a1`
- continuation: `20260727-official-students-cont50-s0-a1`
