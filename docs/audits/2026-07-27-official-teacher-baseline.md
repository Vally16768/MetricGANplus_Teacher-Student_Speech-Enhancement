# Official MetricGAN+ teacher baseline audit

Status: observed diagnostic, not a promoted project experiment.

## Question

The first local full attempt reported teacher test PESQ-WB 2.5290, while the
MetricGAN+ publication and SpeechBrain recipe report approximately 3.15. This
audit determines whether the dataset/evaluator or the local teacher
implementation caused the gap.

## Primary reference

- paper: `https://arxiv.org/abs/2104.03538`;
- official recipe:
  `https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN`;
- official model card:
  `https://huggingface.co/speechbrain/metricgan-plus-voicebank`.

The imported generator is pinned to Hugging Face revision
`a196ce26b3bdace6fa1d819017584bdbcce462a8`. Its
`enhance_model.ckpt` SHA-256 is
`147bfb866bac8264603546e035bf283370e716ed2f4b7412d308d2bcee88304f`.

## Reconciliation

The former local teacher was not the published recipe:

1. it trained only 10 epochs from scratch;
2. its `T0` objective was a simplified enhancement loss, not the alternating
   MetricGAN+ discriminator/history recipe;
3. it used FFT/hop/window 512/160/320 with a Hann window and square-root
   magnitude, while the official generator uses 512/256/512 with a Hamming
   window and `log1p` magnitude followed by `expm1` reconstruction.

The official SpeechBrain adapter evaluated on the frozen 824-pair test manifest
produced:

| Support | PESQ-WB | STOI | SI-SDR | delta-SNR |
|---:|---:|---:|---:|---:|
| 824 | 3.1225 | 0.9311 | 8.4588 | -8.7393 |

The PESQ difference from the former local teacher is +0.5935. The result is
close to, but is not asserted to reproduce exactly, the published 3.15 because
the local manifest/evaluator and current dependency versions are not the
original SpeechBrain test environment. The adapter also peak-normalizes its
output; therefore its energy-sensitive metrics are recorded as guardrails and
must not be interpreted as optimized targets.

The exact local generator reconstruction loads 21/21 tensors with zero skipped
keys and contains 1,895,514 parameters. On the frozen four-pair smoke test it
produced PESQ-WB 3.3407, STOI 0.9334, SI-SDR 9.1924 and delta-SNR -0.9245.
That support is wiring evidence only.

## Corrective design

The canonical T0 teacher is now the pinned official checkpoint at epoch 0.
It first produces separate WB/NB local caches and trains S0-WB/S0-NB. Teacher
fine-tuning then starts from the same T0 package for both control and
PESQ-proxy branches. A branch must improve true `val_select` PESQ-WB while
passing STOI/SI-SDR guardrails before a second cache and fresh S1 students are
scientifically promotable.

No test score participates in branch selection. No source audio, manifest or
dataset file was modified.
