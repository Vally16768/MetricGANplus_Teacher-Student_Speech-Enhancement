# Final T1 current-output calibration audit

Evidence status: **reproduced negative result; downstream work stopped**.

## Outcome

The SpeechBrain MetricGAN discriminator did not pass the predeclared
current-output calibration gate after the permitted retry. The generator
remained frozen and received zero updates. Consequently there is no T1
teacher, no C1 cache and no S1-WB/S1-NB training or comparison.

The execution status `pilot-passed` means that the diagnostic completed and
its package reconciled. It does not mean that the scientific calibration gate
passed; both strict refresh gates are false and the run is non-promotable.

## Frozen protocol

- dataset: external read-only VoiceBank+DEMAND;
- source commit: `21af6b5c4604c1e1c1d18adb4cee1b49b99401ec`;
- E0 source: `20260727-converged-s0-baseline-v2`;
- E0 checkpoint SHA-256:
  `5ece6fbd1ac16cca6df11ea724fb5e3710d6611049f54bbc8d126c79dbbc65d8`;
- profile: WB, 16 kHz, PESQ-WB;
- each refresh: 100 D-update outputs plus 100 disjoint held-out outputs;
- order: current clean/enhanced/noisy, historical enhanced, current again;
- gate: normalized MAE ≤ 0.06, Pearson/Spearman ≥ 0.80, sufficient variance,
  count ≥ 100 and prediction range within raw-PESQ tolerance 0.30.

## Strict calibration results

| Run/refresh | Held-out | Raw MAE | Normalized MAE | Pearson | Spearman | Prediction range | Target range | Gate |
|---|---:|---:|---:|---:|---:|---|---|---|
| A2 / 1 | 100 | 0.9838 | 0.1968 | 0.5379 | 0.5504 | 0.8945–4.8650 | 1.6346–4.4011 | fail |
| A3 / 1 | 100 | 0.9541 | 0.1908 | 0.5334 | 0.5406 | 0.8932–4.8319 | 1.6346–4.4011 | fail |
| A3 / 2 | 100 | 1.0663 | 0.2133 | 0.5545 | 0.5435 | 0.7588–5.2196 | 1.5041–3.9593 | fail |

The warm-start proxy looked substantially better on its static validation
records (Pearson 0.8564, Spearman 0.8737), but that fidelity did not transfer
to held-out outputs from the current E0 generator. The second refresh reduced
the internal D update MSE while held-out raw MAE worsened from 0.9541 to
1.0663. This is direct evidence that training loss was not a safe surrogate
for current-output calibration.

## Integrity and stop decision

The independent A3 package audit reconciled 2/2 cells, 2/2 model packages and
reported zero issues. The two-refresh history contains no generator updates
and no redundant epoch/final validation. Its approximately 37.3 MiB replay is
Desktop-local, generated-only FP16 evidence with zero audio input copies.

The failed discriminator checkpoint and replay remain in the ignored local
run `20260727-teacher-calibration-pilot-t1-a3`; they are not promoted as
research models. The report, exact code/config commit, E0 hash, thresholds and
aggregate evidence are sufficient to reproduce the negative diagnostic.

Per the frozen plan, relaxing thresholds, tuning from test, launching E1/E2,
or training S1 after this failure is prohibited. The canonical publishable
result remains the S0 baseline v2. A future T1 formulation requires a new
predeclared experiment and cannot be relabeled as continuation of this trial.
