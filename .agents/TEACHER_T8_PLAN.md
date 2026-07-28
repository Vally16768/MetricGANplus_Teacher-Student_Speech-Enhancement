# T8 train-only adaptive teacher-routing plan

Status: **complete negative outcome — oracle ceiling gate**

## Cause and hypothesis

T5 and T7 both generalized, but applying one suppression rule to every
utterance gained only `+0.005075` and `+0.004931` PESQ-WB while consuming
almost the full SI-SDR budget. The next capacity must be utterance-adaptive.

T8 retains the official T0 output and the frozen selected T7 output. A compact
ridge router predicts the true-PESQ difference `T7 - T0` from clean-free
features computed from the noisy signal, official logits, mask confidence and
the bounded T7 correction. At inference it selects T7 only when predicted gain
exceeds a train/calibration-frozen threshold; otherwise it returns exact T0.
The clean reference is never an inference input.

## Frozen support and model

- dataset: VoiceBank+DEMAND only, external and read-only;
- teacher/profile: official MetricGAN+ WB at 16 kHz;
- candidate: immutable T7 selection
  `low=-0.30`, `high=0`, `threshold=0`, `temperature=1.5`;
- fit: T3 train identities 576–831, 256 examples;
- calibration: T3 train identities 832–959, 128 examples;
- fit/calibration remain pair- and clean-utterance-disjoint;
- features and regression parameters are serialized in the teacher checkpoint;
- test, teacher cache and students remain blocked.

The feature schema is fixed before label generation:

1. noisy log RMS;
2. log-magnitude mean and standard deviation;
3. normalized spectral centroid;
4. logit mean, standard deviation and quartiles;
5. base-mask mean, standard deviation, low/high-confidence fractions;
6. T7 correction mean and standard deviation;
7. base/T7 mask disagreement.

Fit standardized ridge regression with fixed lambdas
`[0.001, 0.01, 0.1, 1, 10]` and deterministic five-fold train-only
cross-validation. On calibration, select from fixed predicted-gain thresholds
`[0, 0.0025, 0.005, 0.01, 0.02]`.

## Pre-validation gates

Before reading `val_rank`:

- the calibration oracle must gain at least `+0.015` PESQ-WB over T0;
- the learned router must gain at least `+0.005` PESQ-WB on calibration;
- learned-router STOI loss must be at most `0.0015`;
- learned-router SI-SDR loss must be at most `0.15` dB;
- checkpoint reconstruction must reproduce routing decisions and waveforms.

Failure stops T8 before validation and preserves only negative support
evidence.

## Validation and promotion

If pre-validation passes, evaluate the frozen router once on `val_rank`.
Advance to one `val_select` evaluation only if it beats T0 on `val_rank` and
passes the unchanged final STOI/SI-SDR guards.

The final teacher gate is unchanged:

- PESQ-WB gain at least `+0.01` on `val_select`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25` dB;
- non-zero deployable routing;
- no test access.

A passing single search still requires independent recomputation, three
declared support seeds, a paired bootstrap PESQ interval excluding zero,
checkpoint/hash audit and package audit before promotion and shutdown.

## Observed outcome

`20260728-t8-router-wb-s3003-a1` trained on 256 and calibrated on 128
train-only examples. The learned threshold selected T7 for 75/128 utterances
and gained `+0.009197` PESQ, with STOI `-0.000957` and SI-SDR `-0.145724` dB.
The exact oracle gained only `+0.014197`, below the frozen `+0.015` gate, so
T8 correctly stopped before `val_rank`; validation and test were unread. T9
expands routing to four suppression actions.
