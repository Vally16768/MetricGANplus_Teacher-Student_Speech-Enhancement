# T13 train-only multi-objective router plan

Status: **predeclared — implementation validation**

## Cause and hypothesis

T10, T11 and T12 all generalized to approximately `+0.008` PESQ on
`val_select`. T12 used `val_rank` correctly, but its global action-strength
penalty selected a policy at the SI-SDR boundary and still missed the PESQ
gate. Global strength is therefore an insufficient proxy for utterance-level
auxiliary risk.

T13 learns three train-only delta predictors per action: PESQ, STOI and
SI-SDR. A weighted multi-objective utility is folded exactly into the existing
linear deployable router. No clean reference or objective metric is needed at
inference.

## Frozen protocol

- official T0 WB/16 kHz and the four T9 actions remain unchanged;
- fit support is the union of T9 fit/calibration, T10 and T11 support:
  `256 + 128 + 128 + 72 = 584` train-only, pair-deduplicated examples;
- each action receives independent five-fold ridge fits for exact PESQ, STOI
  and SI-SDR deltas;
- utility:
  `pred_PESQ + w_stoi*pred_STOI + w_sisdr*pred_SISDR
  - penalty*abs(low)^2`;
- `w_stoi`: `[0, 1, 2, 4]`;
- `w_sisdr`: `[0, 0.01, 0.02, 0.04]`;
- strength penalty: `[0, 0.01, 0.02]`;
- activation margin: `[0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015]`;
- all 336 policies are ranked from one exact `val_rank` action pass;
- selection requires rank PESQ `>= +0.01`, STOI `>= -0.002`, SI-SDR
  `>= -0.25`, then maximizes exact PESQ;
- one frozen checkpoint is evaluated once on `val_select`;
- final gate is unchanged and test remains unread.

## Exit handling

A passing single run is only a candidate and requires independent
recomputation, confirmations, paired bootstrap and package audit. A failure
blocks caches/students and activates only a separately predeclared successor.
