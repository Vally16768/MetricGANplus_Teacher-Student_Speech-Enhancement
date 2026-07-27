# Converged official-teacher S0 baseline — true-length evaluation

Evidence status: **reproduced and independently audited**.

The selected model files and training decisions are identical to v1. This
package corrects variable-length evaluation by running every utterance through
the model at its true length, so right-padding cannot change bidirectional
teacher context.

## Epoch-20 versus converged selection

| Cell | Protocol | Epoch-20 | Converged | Delta | Stop |
|---|---|---:|---:|---:|---|
| S0-WB | PESQ-WB | 2.596915 | 2.603164 | +0.006248 | best 34, early stopping at 42 |
| S0-NB | PESQ-NB | 3.192184 | 3.216944 | +0.024760 | best 41, early stopping at 49 |

## Corrected profile-matched metrics

| Cell | Split | PESQ | STOI | SI-SDR | Delta-SNR | Support |
|---|---|---:|---:|---:|---:|---:|
| T0-WB-OFFICIAL | val_rank | 2.794723 | 0.869161 | 5.527318 | -0.179092 | 128 |
| T0-WB-OFFICIAL | val_select | 2.785350 | 0.868438 | 5.414771 | -0.548029 | 1690 |
| T0-WB-OFFICIAL | test | 3.130948 | 0.931896 | 8.579135 | -0.797395 | 824 |
| S0-WB | val_rank | 2.598225 | 0.856921 | 5.653052 | 0.238655 | 128 |
| S0-WB | val_select | 2.603164 | 0.854977 | 5.534107 | -0.153790 | 1690 |
| S0-WB | test | 3.051914 | 0.929624 | 9.049854 | -0.156585 | 824 |
| S0-NB | val_rank | 3.197955 | 0.855785 | 5.755264 | 0.315994 | 128 |
| S0-NB | val_select | 3.216944 | 0.853526 | 5.598627 | -0.114961 | 1690 |
| S0-NB | test | 3.615133 | 0.929400 | 9.070931 | -0.141880 | 824 |

WB and NB PESQ use different bandwidth protocols and are never pooled or
directly ranked. Both students stopped through early stopping and neither
selected the epoch-50 ceiling. This is one-seed evidence.
