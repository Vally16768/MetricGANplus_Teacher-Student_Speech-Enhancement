# T9 multi-action adaptive teacher-routing plan

Status: **complete negative outcome — auxiliary-risk gate**

## Cause and hypothesis

T8 proved that clean-free utterance routing generalizes on train-only
calibration: the learned T0/T7 router gained `+0.009197` PESQ-WB with tighter
STOI/SI-SDR losses than the global T7 transform. T8 stopped before validation
because its single-candidate oracle ceiling was only `+0.014197`, below the
predeclared `+0.015` gate.

T9 retains the successful routing principle but expands the action set. For
each utterance it predicts true-PESQ gain for four frozen confidence
corrections and chooses the best predicted positive action or exact T0:

```text
low in [-0.20, -0.40, -0.60, -0.80]
high = 0, threshold = 0, temperature = 1.5
```

This allows weak suppression for sensitive utterances and stronger suppression
only where the predicted benefit justifies it. No clean reference is used at
inference.

## Frozen support and fitting

- dataset/profile: VoiceBank+DEMAND, WB/16 kHz, external read-only;
- base: pinned official T0 checkpoint;
- fit: T3 train identities 576–831, 256 examples;
- calibration: first 128 identities from the T3 `calibration` partition;
- fit/calibration are pair- and clean-utterance-disjoint;
- feature schema: the same reviewed 16 clean-free T8 features, computed once
  per action;
- model: one standardized ridge regressor per action;
- lambdas: `[0.001, 0.01, 0.1, 1, 10]`, deterministic five-fold fit CV;
- decision margin thresholds: `[0, 0.005, 0.01, 0.015, 0.02]`;
- test, cache and students remain blocked.

## Pre-validation gates

Before reading `val_rank`:

- multi-action calibration oracle gain at least `+0.025` PESQ-WB;
- learned router gain at least `+0.010` PESQ-WB;
- learned-router STOI loss at most `0.0015`;
- learned-router SI-SDR loss at most `0.15` dB;
- at least two non-base actions are selected on calibration;
- checkpoint round-trip reproduces the multi-action configuration.

If these fail, T9 stops before validation.

## Validation and promotion

The frozen router is evaluated once on `val_rank`. It reaches one
`val_select` evaluation only when PESQ beats T0 and STOI/SI-SDR pass their
final guards. The final teacher gate remains:

- PESQ-WB gain at least `+0.01`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25` dB;
- no test read.

A single pass still requires independent recomputation, three declared support
seeds, paired bootstrap confidence interval excluding zero, checkpoint/hash
audit and independent package audit before promotion and shutdown.

## Observed outcome

`20260728-t9-router-wb-s3003-a1` completed 256 fit and 128 fresh calibration
examples. The four-action oracle gained `+0.031116` PESQ. The PESQ-only learned
router gained up to `+0.022907`, but lost `0.002732` STOI and `0.450111` dB
SI-SDR. At threshold `0.02` it retained `+0.017368` PESQ with STOI
`-0.001773`, but SI-SDR remained `-0.279120` dB. T9 therefore stopped before
validation and test. T10 freezes the learned regressors and calibrates a more
conservative margin on fresh train-only support.
