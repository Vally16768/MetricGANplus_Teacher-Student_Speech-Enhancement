# T6 true-PESQ affine-logit calibration plan

Status: **complete negative outcome — selected exact T5**

## Cause and hypothesis

T5 generalized a smooth frequency bias across fit, calibration, `val_rank` and
`val_select`, but gained only `+0.005075` PESQ-WB and consumed almost the full
SI-SDR guard (`-0.245701` dB). Making the additive curve more negative is not
safe.

T6 adds a global logit temperature:

```text
new_logit(f,t) = scale * original_logit(f,t) + frequency_curve(f)
```

Scaling can separate high-confidence speech bins from low-confidence noise
bins, unlike a purely additive bias. It folds exactly into the final linear
weight/bias and needs no runtime wrapper.

## Frozen protocol

- support: T3 train identities 192–287 for fit and 288–383 for calibration;
- curves: T5 sweep 1 and T5 sweep 3 only;
- scales: `[0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]`;
- evaluate all 14 combinations on fit with hard STOI/SI-SDR guards;
- evaluate the best five fit-safe combinations on disjoint calibration;
- evaluate the best three calibration-safe combinations plus T0 on
  `val_rank`;
- evaluate one selected candidate on `val_select`;
- never read test or train students.

Selection and promotion keep the unchanged `+0.01` PESQ, `-0.002` STOI and
`-0.25` dB SI-SDR gate. A single-seed pass still requires independent
re-evaluation, three declared seeds and paired bootstrap before shutdown.

## Observed outcome

`20260728-t6-affine-wb-s3003-a1` completed the clean 14/5/3 funnel on fresh
96/96 support. It selected T5 sweep 3 with scale `1.0`, producing exactly the
T5 `val_select` deltas: PESQ `+0.005075`, STOI `-0.000782` and SI-SDR
`-0.245701` dB. The gate failed and test remained unread. T7 replaces global
calibration with confidence-conditioned logits.
