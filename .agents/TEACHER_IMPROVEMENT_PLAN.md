# Teacher improvement and TTS-transfer plan

Status: **active — P4 implementation and validation**.

Owner: the MetricGAN+ research campaign.

Activation gate satisfied by baseline release commit `e6388d4`: P1–P3 in
`.agents/EXECUTION_TODO.md` passed, including the merged S0 audit, resume
repair and sanitized baseline promotion. This plan does not retroactively
alter or replace S0 evidence.

## Research questions

### Enhancement question

Can the pinned official WB MetricGAN+ generator be improved by a
current-output-calibrated metric discriminator, while preserving STOI and
SI-SDR, and does an accepted teacher improvement transfer to both a WB and an
NB student?

### TTS question

Can the differentiable metric-critic interface later improve a selected TTS
generator when the critic is trained and calibrated on synthesis outputs?

The TTS question is a separate future campaign. VoiceBank+DEMAND enhancement
results are not evidence for a TTS claim.

## Scientific hypothesis

The official T0 teacher is already a MetricGAN+ generator. Therefore, merely
adding another discriminator is not the intervention. The enhancement
intervention is to keep the SpeechBrain MetricGAN discriminator faithful to
the distribution produced by the current generator and to block generator
updates whenever that fidelity is insufficient.

The primary hypothesis is:

> A dynamically refreshed, current-output-calibrated PESQ discriminator can
> increase true `val_select` PESQ-WB by at least `0.01` relative to the pinned
> T0 checkpoint, without reducing STOI by more than `0.002` or SI-SDR by more
> than `0.25 dB`.

## Frozen experimental boundaries

- Dataset: VoiceBank+DEMAND only, external and read-only.
- Teacher profile: WB, 16 kHz.
- Teacher initialization: the pinned official SpeechBrain MetricGAN+
  checkpoint.
- Student profiles after an accepted teacher: WB/16 kHz with PESQ-WB and
  NB/8 kHz with PESQ-NB.
- Selection split: frozen `val_select`.
- Test split: reporting only, after model selection.
- Runtime: shared Desktop virtual environment and CUDA for all training.
- Caches and discriminator replay: ignored Desktop-local storage; never under
  the dataset root and never in Git.
- No S1 cache or student training begins unless the teacher gate passes.

## Enhancement architecture

```text
external noisy WB ---------------------------> T1 generator G
                                                     |
                                                     v
external clean WB --------------------------> current enhanced
        |                                            |
        +--> true PESQ-WB label ---------------------+
        |                                            v
        +-----------------------------> metric discriminator D
                                                     |
                           calibration gate <--------+
                                                     |
                                  if passed: freeze D
                                                     |
                      PESQ target + T0 anchor + spectral/SI-SDR guards
                                                     |
                                                     v
                                              generator update
```

True PESQ produces labels and evaluation evidence; it is not differentiated
through. D supplies the differentiable generator signal.

## Experiment matrix

| ID | Branch | Purpose |
|---|---|---|
| `E0-T0` | official teacher, frozen | immutable reference |
| `E1-CONTROL` | conservative T0 fine-tuning without a new metric term | isolate ordinary fine-tuning |
| `E2-PESQ` | same fine-tuning plus dynamically refreshed PESQ D | test the primary hypothesis |
| `E3-PESQ-LR` | one predeclared alternative teacher LR | only if E2 is safe but inconclusive |
| `E4-MULTI` | later multi-objective ablation | only after PESQ-only passes |

E1 and E2 use the same initialization, seed, support and schedule. E3 may test
`3e-6` only after the primary `1e-6` configuration. Do not introduce a broad
hyperparameter search or tune from test results.

## Stage 1 — Calibration-only diagnostic

Do not update G in this stage.

1. Generate at least 100 current T0 outputs per discriminator refresh.
2. Combine clean, noisy, current enhanced and historical enhanced examples.
3. Stratify the sampled support where possible by true PESQ, SNR, speaker and
   noise condition.
4. Keep a held-out current-generator calibration partition.
5. Execute the canonical current/history/current D refresh.
6. Record raw and normalized MAE/RMSE, Pearson, Spearman, score coverage and a
   predicted-versus-true calibration plot.

The implementation draws 100 discriminator-update records plus a disjoint 100
record held-out current-output partition. Held-out examples never enter the
current/history/current discriminator updates.

The initial predeclared calibration gate is:

- at least 100 held-out current outputs;
- normalized PESQ MAE at most `0.06` (raw-PESQ equivalent `0.30`);
- Pearson at least `0.80`;
- Spearman at least `0.80`;
- no near-constant output or current-score extrapolation beyond the calibrated
  range.

If the gate fails, refresh D again or stop the trial. Do not update G.

## Stage 2 — Teacher-only pilot

Use a conservative generator objective:

```text
L_G =
  lambda_pesq * MSE(D(clean, G(noisy)), 1)
  + lambda_anchor * L_wave(G(noisy), T0(noisy))
  + lambda_spectral * L_log_spectral(G(noisy), clean)
  + lambda_sisdr * L_SI-SDR(G(noisy), clean)
```

Operational order for every generator epoch:

1. generate current outputs and calculate true PESQ-WB labels;
2. update local generated-only replay;
3. refresh D on current/history/current examples;
4. evaluate D on held-out current outputs;
5. skip the G update if the calibration gate fails;
6. otherwise freeze D and perform one G epoch;
7. evaluate true PESQ-WB, STOI, SI-SDR and delta-SNR on `val_select`;
8. save the complete resumable state after evaluation;
9. apply LR reduction and early stopping from `val_select` only.

Initial pilot limits:

- teacher LR `1e-6`;
- maximum 10 generator epochs;
- early stopping after three non-improving accepted evaluations;
- retain the official T0 trust anchor;
- record generator drift from T0 and predicted-versus-true PESQ disagreement.

Before launch, verify by test that a resume restores G, D, both optimizers,
scheduler, early-stopping state, replay identity and post-evaluation history.

## Stage 3 — Teacher promotion

Promote T1 only when all conditions hold:

- true `val_select` PESQ-WB gain is at least `+0.01` over E0;
- STOI loss is no greater than `0.002`;
- SI-SDR loss is no greater than `0.25 dB`;
- discriminator calibration passed before every accepted G update;
- E2 beats E1, so the effect is attributable to the metric intervention;
- package provenance, metrics, plots and checkpoint hashes reconcile in an
  independent audit.

Only after selection may the accepted checkpoint be evaluated once on test.
A failed or inconclusive teacher remains historical negative evidence and
cannot trigger S1 training.

## Stage 4 — Transfer to students

If and only if Stage 3 passes:

1. create a new content-addressed C1 cache from the accepted T1 checkpoint;
2. preserve external noisy/clean inputs and store regenerable FP16 teacher
   targets locally;
3. train fresh S1-WB and S1-NB students;
4. match the S0 architecture, seeds, schedule, LR policy and stopping rule;
5. evaluate each student against its bandwidth-matched reference and PESQ
   mode;
6. report paired `T0→S0` versus `T1→S1` deltas with uncertainty and
   per-condition slices.

## Stage 5 — Optional multi-objective enhancement ablation

Do not mix this stage into the primary PESQ experiment. After E2 passes, test
one change at a time:

- direct differentiable SI-SDR regularization;
- a separately normalized metric head;
- broader discriminator score support inspired by MetricGAN+/-.

STOI remains an evaluation guard initially. A single uncalibrated scalar that
mixes PESQ, STOI and SI-SDR is not allowed.

## Separate TTS campaign

Modern neural TTS systems already use waveform adversarial discriminators.
The proposed addition is a synthesis-domain metric critic, not reuse of the
VoiceBank enhancement discriminator.

Before a TTS experiment:

1. select and freeze the TTS generator baseline and its dataset contract;
2. define a distinct run namespace, configs, splits and claims;
3. generate a representative synthesis-output calibration set;
4. prefer synthesis-appropriate targets: human MOS/naturalness,
   ASR intelligibility, speaker similarity and prosody/duration measures;
5. use PESQ only in a specifically justified, time-aligned paired ablation;
6. calibrate the critic on held-out outputs from the current TTS generator;
7. compare a no-critic control with the metric-critic branch;
8. require objective evaluation plus listening/MOS evidence before a TTS
   improvement claim.

PESQ alone is not the default TTS objective because synthesis permits valid
variation in timing, prosody and realization relative to one recorded
reference.

## Execution checklist

```text
C41 completion and audit
-> freeze T0/S0 evidence
-> implement teacher-only calibration command
-> resume-state and calibration tests
-> full test suite + project guard
-> clean committed snapshot
-> calibration-only diagnostic
-> CUDA smoke
-> monitored teacher-only pilot
-> E0/E1/E2 true-metric audit
-> teacher promotion gate
-> only on pass: C1 + S1-WB + S1-NB
-> independent report and promotion audit
-> later, separate TTS campaign
```

## Primary methodological references

- MetricGAN: <https://proceedings.mlr.press/v97/fu19b.html>
- MetricGAN+: <https://arxiv.org/abs/2104.03538>
- Official SpeechBrain MetricGAN+ model card:
  <https://huggingface.co/speechbrain/metricgan-plus-voicebank>
- MetricGAN+/-: <https://arxiv.org/abs/2203.12369>
- HiFi-GAN: <https://arxiv.org/abs/2010.05646>
- VITS: <https://arxiv.org/abs/2106.06103>
