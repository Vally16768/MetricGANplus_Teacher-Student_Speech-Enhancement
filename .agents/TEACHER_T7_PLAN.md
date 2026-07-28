# T7 confidence-conditioned true-PESQ calibration plan

Status: **predeclared — implementation pending**

## Cause

T5 improved `val_select` PESQ-WB by `+0.005075`, but its additive frequency
curve reduced SI-SDR by `0.245701` dB. T6 tested whether a global logit scale
could separate speech and noise; selection returned the exact T5 solution at
scale `1.0`. Global calibration has exhausted its useful capacity before the
`+0.01` teacher gate.

T7 changes the functional form, not the dataset or evaluation protocol. For
each time-frequency mask logit `z`, it applies:

```text
z' = z + low + (high - low) * sigmoid((z - threshold) / temperature)
```

Low-confidence bins receive stronger suppression while high-confidence speech
bins receive little suppression or a small positive correction. The operation
is deterministic, uses no clean reference at inference, and is serialized in
the ordinary teacher checkpoint configuration.

## Frozen protocol

- teacher: pinned official MetricGAN+ WB checkpoint, 16 kHz;
- dataset: VoiceBank+DEMAND only, external and read-only;
- support: T3 train identities 384–479 for fit and 480–575 for calibration;
- fit/calibration: 96/96 pair- and clean-utterance-disjoint examples;
- true objective: PESQ-WB;
- auxiliary guardrails: STOI and SI-SDR on every funnel stage;
- `val_rank`: rank only completed, eligible configurations;
- `val_select`: evaluate exactly one selected configuration;
- test: unread during search and selection;
- cache/student work: blocked until promotion.

The fixed candidate grid is:

- `low`: `-0.20`, `-0.30`, `-0.40`;
- `high`: `0.00`, `+0.05`;
- `threshold`: `-6.0`, `-4.0`, `-2.0`, `0.0`;
- `temperature`: `1.5`.

This gives 24 predeclared candidates. Retain the top eight fit-safe candidates,
then the top four calibration-safe candidates, then the top two
`val_rank`-safe candidates. T0 participates in ranking as the fallback.

## Gate and confirmation

Promotion thresholds remain unchanged:

- PESQ-WB gain at least `+0.01` on `val_select`;
- STOI loss no greater than `0.002`;
- SI-SDR loss no greater than `0.25` dB;
- non-zero deployable calibration;
- no test access.

A single passing search is only a candidate. Before promotion and shutdown,
repeat an independent deterministic evaluation, confirm the selected method
across three predeclared support seeds, require a paired bootstrap PESQ
confidence interval excluding zero, verify checkpoint round-trip and hashes,
and run the independent package audit.

If T7 fails, preserve its concise negative evidence and predeclare a successor
with genuinely new capacity. Do not relax the gate and do not train students.
