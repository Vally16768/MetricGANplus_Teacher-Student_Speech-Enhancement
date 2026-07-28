# T15 cross-fitted quadratic calibration plan

Status: **predeclared — implementation pending**

## Cause and hypothesis

T14 improved the best safe `val_select` PESQ gain to `+0.009365`, only
`0.000635` below the gate. Quadratic fits increased train correlation from
approximately `0.49` to `0.55–0.57`, but their in-sample predictions still
select a policy near the SI-SDR boundary.

T15 retains the exact T14 feature map, support, actions and final gate. It
produces five-fold out-of-fold train predictions for each action/metric, fits
only an affine shrinkage calibration from those OOF predictions to true
train-only deltas, and folds that calibration into the final full-support
quadratic ridges. This targets prediction calibration without using
`val_rank`, `val_select` or test labels during model fitting.

## Frozen protocol

- official T0, four T9 actions, 584 train-only examples;
- exact 152-feature quadratic transform and T14 lambda grid;
- deterministic five-fold out-of-fold predictions;
- per-action/per-metric affine calibration with non-negative slope clipped to
  `[0, 1.5]`;
- final full-support quadratic ridge receives the frozen affine correction;
- same 336 T13/T14 policy family ranked once on `val_rank`;
- one frozen checkpoint evaluated once on `val_select`;
- final gate unchanged; test unread.

Passing still requires independent recomputation, confirmation, paired
bootstrap and package audit before promotion or shutdown.
