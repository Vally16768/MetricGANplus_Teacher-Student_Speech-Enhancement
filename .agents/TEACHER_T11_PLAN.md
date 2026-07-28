# T11 risk-penalized multi-action routing plan

Status: **executed — final PESQ gate failed**

## Cause and hypothesis

T10 selected a conservative `0.025` activation margin and passed both
auxiliary guards on `val_select`, but gained only `+0.008015` PESQ-WB. Its
router still chose only the strongest `low=-0.8` action. T11 keeps all frozen
T9 regressors but subtracts an action-strength penalty from each predicted
PESQ gain before choosing the action:

```text
utility(action) = predicted_delta_pesq(action) - lambda * abs(low)^2
```

This makes weak/moderate suppression competitive for marginal utterances
without a clean inference input.

## Frozen search

- VoiceBank+DEMAND WB/16 kHz and official T0 only;
- T9 ridge weights/scales remain frozen; only serialized biases receive the
  declared deterministic penalty;
- fresh calibration: final 72 identities of the T3 `audit` partition, disjoint
  from T9 and T10 supports;
- penalties: `[0.005, 0.01, 0.02, 0.03, 0.04]`;
- utility margins: `[0.005, 0.01, 0.015, 0.02, 0.025]`;
- final gate unchanged: PESQ `>= +0.01`, STOI `>= -0.002`, SI-SDR
  `>= -0.25`, no test read.

Select only an eligible fresh-calibration policy, then evaluate it once on
`val_rank` and conditionally once on `val_select`. A single passing result is
still only a candidate and requires independent recomputation, multiple
declared support seeds, paired bootstrap and package audit before shutdown.

## Outcome

T11 selected penalty `0.04` and margin `0.005`. It gained `+0.014979` PESQ on
fresh calibration and `+0.010068` on `val_rank`, with both auxiliary
guardrails intact. The frozen checkpoint gained only `+0.008349` PESQ on
`val_select` (STOI `-0.001395`, SI-SDR `-0.227965` dB), so the unchanged final
gate failed. Test, cache generation and student training remained unread or
unexecuted.
