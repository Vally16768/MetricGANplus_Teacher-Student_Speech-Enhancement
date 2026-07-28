# T2 teacher-improvement successor plan

Status: **planned — no T2 training started**  
Owner: MetricGAN+ research campaign  
Predecessor: `TEACHER_IMPROVEMENT_PLAN.md` / T1, preserved as negative evidence  
Execution board: `TEACHER_SUCCESSOR_TODO.md`

## Outcome sought

Improve the pinned official WB MetricGAN+ teacher over T0 on true,
bandwidth-correct VoiceBank+DEMAND metrics, then retrain the WB and NB students
only if the teacher improvement is real and attributable to the metric
intervention.

The promotion hypothesis remains:

> T2 improves true `val_select` PESQ-WB by at least `+0.01` over T0, loses no
> more than `0.002` STOI and no more than `0.25 dB` SI-SDR, and beats a
> schedule-matched non-metric fine-tuning control.

## Why T1 failed

T1 stopped correctly before any generator update. Its final discriminator
calibration on current T0 outputs had:

- normalized MAE `0.2133`, versus the required maximum `0.06`;
- Pearson `0.5545`, versus the required minimum `0.80`;
- Spearman `0.5435`, versus the required minimum `0.80`;
- predicted raw-PESQ range `0.7588–5.2196`, outside the held-out target range
  `1.5041–3.9593`.

The internal discriminator update loss decreased while held-out current-output
fidelity worsened. This means the immediate problem is not teacher learning
rate or the generator objective. The current discriminator does not yet
provide a trustworthy local surrogate around the official teacher.

T1 evidence and terminology remain immutable. This successor uses the T2/D2
namespace.

## Evidence-based corrections

The official SpeechBrain recipe uses:

- the published four-convolution MetricGAN discriminator;
- normalized PESQ labels `(PESQ + 0.5) / 5`;
- separate clean, enhanced and noisy discriminator updates;
- batch size `1`;
- `100` sampled training utterances per epoch;
- current → historical → same-current discriminator passes;
- `20%` historical replay;
- generator and discriminator learning rates of `5e-4` in the original
  from-scratch recipe.

Our alternating update order matches the official high-level sequence.
However, the T1 warm-start proxy was trained in padded batches larger than one
and was allowed only two fixed-generator refreshes before the strict gate.
T2 must first test exact feature/model/update parity and then fit D to
convergence on a disjoint, score-covered T0-output support.

MetricGAN+/- reports that predictor robustness is limited by the score
distribution observed during training. T2 therefore predeclares a
score-support expansion only as a second discriminator ablation, after the
exact-recipe D2 branch has been measured.

## Frozen boundaries

- Dataset: VoiceBank+DEMAND only; external and read-only.
- Teacher: WB, 16 kHz, initialized from the pinned official T0 checkpoint.
- Frequent diagnostics: frozen `val_rank`.
- Final candidate selection: frozen `val_select`.
- Test: reporting only after selection; it cannot tune T2.
- Training: CUDA from the shared Desktop virtual environment.
- Replay, generated candidates and metric caches: ignored Desktop-local
  storage, never the dataset root and never Git.
- No dataset audio is copied into Git.
- No C2 or student training starts before the T2 promotion gate passes.

## Experiment matrix

| ID | Intervention | Purpose |
|---|---|---|
| `E0-T0` | pinned official teacher, frozen | immutable baseline |
| `D2-OFFICIAL` | D from scratch, exact official architecture/update semantics | primary discriminator fidelity test |
| `D2-RANGE` | D2 plus predeclared score-support widening | conditional MetricGAN+/- inspired ablation |
| `E1-CONTROL` | conservative teacher fine-tuning without metric loss | isolate ordinary fine-tuning |
| `E2-METRIC` | same schedule plus accepted D2 metric loss | primary T2 intervention |
| `E3-LR` | one predeclared alternative generator LR | only if E2 is safe but inconclusive |

No broad hyperparameter search is allowed. `D2-RANGE` cannot replace a failed
`D2-OFFICIAL` result silently; both outcomes must be reported.

## Stage A — Exact official parity audit

Do not train G.

1. Pin the official SpeechBrain source revision used for parity.
2. Build a fixed-tensor parity fixture for:
   - Hamming STFT `512/256/512`;
   - `log1p(sqrt(abs(STFT)))` features;
   - discriminator architecture and initialization semantics;
   - normalized PESQ labels;
   - clean/enhanced/noisy update order;
   - current/history/current replay selection.
3. Compare local and official discriminator outputs after importing the same
   state dict.
4. Add batch-size and right-padding invariance checks. D2 training must use
   batch `1`; evaluation must trim every utterance to its true length.
5. Verify that D gradients are disabled during evaluation and G updates cannot
   begin when a D gate fails.

Gate A passes only with numerical parity within the declared tolerance and
with all invariance/control-flow tests passing.

## Stage B — Build a fixed D2 support

Do not train G.

Create train/calibration/audit partitions from the existing VoiceBank training
support. The canonical content-addressed staging did not retain the original
VoiceBank speaker IDs, so exact speaker-disjointness cannot be reconstructed
without inventing metadata. T2 therefore enforces strict pair/clean-utterance
disjointness and records the missing speaker/noise metadata as a limitation.
The initial target sizes are:

| Partition | Current T0 utterances | Use |
|---|---:|---|
| D train | 1,000 | D updates only |
| D calibration | 200 | scheduler and early stopping |
| D audit | 200 | one-shot D2 gate |

For each selected utterance retain labels for clean, noisy and current T0
enhanced speech. Store only regenerable candidates locally in FP16; retain
source IDs, T0 hash, manifest hash, PESQ implementation/mode and waveform
length. Never cache or rewrite source noisy/clean audio.

Before training, plot the PESQ distribution by candidate type, estimated input
SNR and every condition retained by the source manifest. Partitions must be
disjoint by pair and clean utterance, and the audit partition must not
influence optimization. Speaker/noise-condition slices are reported only if
their source identities are recoverable.

## Stage C — Fit and audit `D2-OFFICIAL`

Train a fresh D with the official architecture and batch-1 update semantics:

- D learning rate `5e-4`;
- current → historical → same-current passes;
- history portion `0.20`;
- maximum `20` fixed-generator D epochs;
- `ReduceLROnPlateau`, factor `0.5`, patience `2`, minimum LR `1e-6`;
- early stopping patience `5` on the calibration composite;
- complete resumable state after every calibration.

Checkpoint selection minimizes the predeclared calibration composite:

```text
normalized_MAE
+ 0.10 * max(0, 0.80 - Pearson)
+ 0.10 * max(0, 0.80 - Spearman)
+ 0.01 * raw_prediction_range_excess
```

This composite controls scheduling, early stopping and D checkpoint selection.
It does not replace or relax the untouched audit thresholds below.

The fixed audit support is evaluated once after D selection. The existing
strict fidelity gate is retained:

- at least `200` audit current outputs;
- raw PESQ MAE at most `0.30` (normalized MAE at most `0.06`);
- Pearson at least `0.80`;
- Spearman at least `0.80`;
- prediction standard deviation at least `0.02`;
- no prediction-range escape beyond `0.30` raw PESQ;
- finite metrics and no material subgroup collapse.

Also require local usefulness around T0. For controlled candidate pairs near
the T0 output:

- predicted and true PESQ deltas must have matching sign in at least `70%` of
  pairs;
- Spearman correlation between predicted and true deltas must be at least
  `0.60`;
- no systematic opposite-gradient behavior in a speaker/noise slice.

This local directional gate is mandatory because G follows D gradients, not
only its global scalar predictions.

## Stage D — Conditional score-support widening

Run only if `D2-OFFICIAL` fails because of insufficient score coverage or weak
local ranking, not because parity is broken.

Create one predeclared `D2-RANGE` support by adding controlled candidate
variants derived only from VoiceBank+DEMAND train pairs:

- interpolation between noisy, T0-enhanced and clean signals;
- bounded mask/output perturbations around T0;
- balanced sampling across raw-PESQ bins.

No test or validation-clean output is used. The architecture, optimizer,
stopping rule and untouched D audit set remain identical to
`D2-OFFICIAL`. This is a lightweight score-support ablation inspired by
MetricGAN+/-; it must not be described as a reproduction of its de-generator.

If D2 still fails the full and local gates, stop. Do not update G.

## Stage E — Controlled teacher pilot

Only an accepted D2 checkpoint may enter this stage.

Use the same T0 initialization, seed, support, optimizer schedule and maximum
accepted epochs for both branches:

```text
E1-CONTROL:
  L = anchor(T0) + log-spectral(clean) + SI-SDR(clean)

E2-METRIC:
  L = E1 losses + lambda_pesq * MSE(D2(G(noisy), clean), 1)
```

Pilot policy:

- primary G learning rate `1e-6`;
- maximum `10` accepted G epochs;
- early stopping patience `3`;
- full G/D/optimizer/scheduler/patience/replay resume state;
- frequent ranking on `val_rank`;
- final candidate comparison on `val_select`;
- reject and roll back an epoch if D fidelity fails, true `val_rank` PESQ
  regresses materially or a guard metric crosses its limit.

`E3-LR=3e-6` may run only if E2 is stable, passes all safety checks and is
positive but below `+0.01`. It is not allowed after a harmful E2 result.

## Stage F — Teacher promotion and confirmation

Promote T2 only when:

- `val_select` PESQ-WB gain over E0 is at least `+0.01`;
- STOI loss is at most `0.002`;
- SI-SDR loss is at most `0.25 dB`;
- E2 beats the matched E1 control;
- D2 passed full and local fidelity before every accepted update;
- the selected result is confirmed over three declared seeds with paired
  uncertainty;
- checkpoint, metrics, histories and source hashes pass independent audit.

Only the selected T2 is evaluated on test. A safe but sub-threshold result is
reported as inconclusive, not promoted.

## Stage G — Student transfer

Only after Stage F passes:

1. build Desktop-local, content-addressed C2 from the accepted T2 hash;
2. train fresh S2-WB and S2-NB from zero;
3. preserve the S0 architecture, declared seeds and max-50
   Reduce-LR/early-stopping policy;
4. evaluate WB with WB reference/PESQ-WB and NB with NB reference/PESQ-NB;
5. report `T2−T0`, `S2-WB−S0-WB` and `S2-NB−S0-NB` with paired uncertainty;
6. promote only audited, sanitized models/reports—not the dataset, replay,
   generated audio or regenerable cache.

## Stop rules

- Parity failure: repair/test; no D2 experiment.
- D2 full or local gate failure: preserve negative evidence; no G update.
- E2 harmful versus E0 or E1: stop; do not try the larger LR.
- Teacher below promotion threshold: no C2/S2.
- Best result at an epoch ceiling: mark `ceiling-limited`; do not extend
  automatically.
- Any test leakage, dataset mutation, personal path or dirty-worktree training
  invalidates promotion.

## Required outputs

- parity test report and pinned official source revision;
- D support manifest hashes and split audit;
- calibration/rank/range/subgroup plots;
- local predicted-versus-true delta plot;
- E0/E1/E2 learning curves and guard metrics;
- selected model and complete training-state hashes;
- independent audit and article-ready negative or positive claim map.

## Primary references

- MetricGAN: <https://proceedings.mlr.press/v97/fu19b.html>
- MetricGAN+: <https://arxiv.org/abs/2104.03538>
- SpeechBrain recipe:
  <https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN>
- SpeechBrain model implementation:
  <https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/MetricGAN.py>
- MetricGAN+/-: <https://arxiv.org/abs/2203.12369>
