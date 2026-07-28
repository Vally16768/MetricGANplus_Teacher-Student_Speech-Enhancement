# T14 quadratic multi-objective router plan

Status: **executed — final PESQ gate failed**

## Cause and hypothesis

T13 improved `val_select` to `+0.008806` PESQ with safe auxiliaries, the best
routed result so far, but remained `0.001194` below the final gate. Its linear
PESQ delta fit was limited (action-1 train correlation approximately `0.49`).
T14 tests whether regularized pairwise interactions between the 16 clean-free
router features improve action ranking without changing the actions, support
or metric protocol.

## Frozen protocol

- exact official T0, four T9 actions and 584 train-only examples;
- deterministic feature map: 16 original features plus all 136 upper-triangle
  pairwise products, total 152;
- five-fold ridge lambda grid `[0.1, 1, 10, 100, 1000]`;
- separate PESQ/STOI/SI-SDR delta fits for every action;
- same predeclared 336 multi-objective policies as T13;
- selection only on `val_rank`; one frozen quadratic checkpoint is evaluated
  once on `val_select`;
- final PESQ/STOI/SI-SDR gate unchanged; test remains unread.

The transform and all fitted normalizers/weights are serialized. Inference
requires noisy speech only and evaluates no objective metric.

## Exit handling

A passing result remains a candidate until independent recomputation,
confirmation, paired bootstrap and package audit. Failure blocks students and
activates only a newly predeclared successor.

## Outcome

T14 selected SI-SDR weight `0.02`, strength penalty `0.02` and margin
`0.0025`. Rank gained `+0.011275` PESQ; `val_select` gained `+0.009365`,
with STOI `-0.001466` and SI-SDR `-0.241138` dB. It is the strongest routed
result so far but remains `0.000635` below the unchanged PESQ gate. Test was
unread.
