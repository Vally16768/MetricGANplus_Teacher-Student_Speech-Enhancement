# T12 rank-selected risk-policy plan

Status: **predeclared — implementation validation**

## Cause and hypothesis

T11 preserved the auxiliary metrics but generalized from `+0.014979` PESQ on
fresh calibration and `+0.010068` on `val_rank` to only `+0.008349` on
`val_select`. Its policy was selected on 72 calibration examples even though
the frozen protocol reserves `val_rank` for configuration ranking.

T12 does not retrain the teacher or the T9 regressors. It performs one exact
four-action metric pass over `val_rank`, ranks a predeclared risk-policy grid,
freezes exactly one deployable checkpoint, and evaluates that checkpoint once
on `val_select`. The final split remains unavailable during selection and test
remains unread.

## Frozen policy family

- source teacher: official MetricGAN+ T0, WB/16 kHz;
- source router: frozen T9 features, four ridge models and actions
  `[-0.2, -0.4, -0.6, -0.8]`;
- utility: `predicted_delta_pesq - penalty * abs(low)^2`;
- penalties:
  `[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]`;
- activation margins:
  `[0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]`;
- selection split: `val_rank` only;
- selection rule: highest exact PESQ gain among policies with PESQ
  `>= +0.01`, STOI `>= -0.002`, SI-SDR `>= -0.25`, and at least one routed
  example; deterministic tie break prefers SI-SDR, STOI, penalty, then margin;
- final gate on one frozen checkpoint and `val_select`: PESQ `>= +0.01`,
  STOI `>= -0.002`, SI-SDR `>= -0.25`;
- test is never read.

The 72-policy multiplicity is reported explicitly. `val_select` is the
independent final check, not another policy-selection surface.

## Exit handling

- fail: preserve T12 as negative evidence, do not generate a cache or train
  students, and predeclare a materially different successor;
- pass: treat the result only as a candidate, then perform independent metric
  recomputation, declared confirmation runs, paired bootstrap and package
  audit;
- shutdown is allowed only after that confirmation establishes a genuine
  improved teacher, or under the separate unplugged-and-battery-below-90%
  safety condition.
