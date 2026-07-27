# Corrected converged S0 baseline promotion audit

Evidence status: **reproduced and independently audited**.

The canonical package is
`experiments/runs/20260727-converged-s0-baseline-v2`. It retains the exact
official T0, S0-WB and S0-NB model hashes from v1 and corrects evaluation to
run every variable-length utterance at its true length.

## Cause and correction

The official teacher contains a bidirectional LSTM. The previous evaluator
right-padded unequal utterances and passed the padded batch through that
network, allowing padding to influence backward recurrent context. A direct
CUDA regression check showed identical PESQ for requested batch sizes 1 and 4
after switching to per-utterance model inference.

## Corrected metrics

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

WB and NB PESQ use separate profile-matched protocols and are not pooled.
The students remain numerically unchanged to within 0.00022 PESQ; the
material correction is the teacher validation score.

## Selection and provenance

| Cell | Model SHA-256 | Selection |
|---|---|---|
| T0-WB-OFFICIAL | `5ece6fbd1ac16cca6df11ea724fb5e3710d6611049f54bbc8d126c79dbbc65d8` | official checkpoint |
| S0-WB | `dc1d2d2171876fb5665bd447506e3371492a4619cc8f2749cbfca7292f1ca335` | best 34, early stop 42 |
| S0-NB | `1b89e6b5931eb3a4bb63db7844ffe5e74486e9bf75b835342926776336d11491` | best 41, early stop 49 |

The private corrective run
`20260727-converged-s0-true-length-a1` reconciled 3/3 cells and 3/3 models
with zero audit issues and was marked promotable. The public package excludes
VoiceBank+DEMAND audio, generated audio, caches, replay buffers and training
states. Its canonical contract, artifact inventory and privacy checks must
pass again after the package is committed.
