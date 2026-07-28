# T3 teacher-improvement plan

Status: **planned — no T3 implementation or training started**  
Owner: MetricGAN+ teacher–student campaign  
Predecessor: T2/D2, closed as negative evidence  
Execution board: `TEACHER_T3_TODO.md`

## Objective

Improve the pinned official WB/16 kHz teacher on VoiceBank+DEMAND without
repeating the failed absolute-PESQ discriminator formulation. T3 separates:

1. stable supervised fine-tuning;
2. a direct differentiable perceptual loss;
3. a conditional pairwise metric critic trained for the local ordering that
   the generator actually uses.

The teacher is promoted only if true `val_select` PESQ-WB improves by at least
`+0.01`, STOI drops by at most `0.002`, SI-SDR drops by at most `0.25 dB`, and
the perceptual branch beats the matched supervised control. Test remains
reporting-only.

## Diagnosis carried from T2

T2 disproved the assumption that better global score coverage is sufficient:

| Critic | nMAE | Pearson | Spearman | local sign | local rho |
|---|---:|---:|---:|---:|---:|
| D2-OFFICIAL | 0.2895 | 0.7626 | 0.7768 | 0.5291 | -0.4929 |
| D2-RANGE | 0.3287 | 0.7283 | 0.7599 | 0.3266 | -0.6221 |

The generator follows a local gradient, not an aggregate correlation.
D2-RANGE widened PESQ support but made the local direction worse. T3 therefore
does not extend D2 epochs, relax its thresholds or reuse either failed
checkpoint for generator guidance.

## Frozen boundaries

- Dataset: VoiceBank+DEMAND only, external and read-only.
- Teacher: pinned official MetricGAN+ WB checkpoint, 16 kHz.
- Students: no training before teacher promotion; later WB/16 kHz and NB/8 kHz.
- Runtime: shared Desktop virtual environment and CUDA for every training run.
- Data roles: `train_fit` optimization, `val_rank` frequent rollback/ranking,
  `val_select` final candidate selection, test reporting only.
- Generated candidates/cache: ignored Desktop-local FP16 storage; never Git or
  the dataset root.
- Existing T0/S0/T1/T2 evidence remains immutable.

## Experiment matrix

| ID | Intervention | Purpose |
|---|---|---|
| `E0-T0` | frozen official checkpoint | immutable teacher baseline |
| `E1-SUP` | trust-region supervised fine-tuning | measures ordinary fine-tuning |
| `E2-PMSQE` | E1 plus direct PESQ-inspired loss | primary low-risk perceptual branch |
| `D3-RANK` | pairwise local PESQ rank critic | conditional learned-metric branch |
| `E3-RANK` | E1 plus frozen accepted D3 pairwise objective | tests the discriminator hypothesis |

`E2-PMSQE` and `E3-RANK` are separate ablations. Neither may be described as
an improvement until true metric gates pass.

## Stage A — Reconcile headroom and implement losses

Do not update the teacher.

1. Re-evaluate the exact T0 hash with true-length WB inference on the frozen
   internal supports; verify the current canonical values and metric code.
2. Pin one reviewed PMSQE implementation and its revision/license. PMSQE is a
   differentiable PESQ-inspired loss, not exact ITU PESQ.
3. Implement:
   - multi-resolution magnitude/log-magnitude STFT loss;
   - SI-SDR loss;
   - output trust-region loss against frozen T0;
   - PMSQE with explicit WB/16 kHz preprocessing.
4. Test finite forward/backward values, true-length behavior, batch/padding
   invariance, amplitude scaling, silence, short utterances and CUDA AMP.
5. Confirm that no new dependency is installed outside the shared venv.

The direct perceptual objective is:

```text
L_E1 = L_MRSTFT(clean, G(x))
     + 0.10 * L_SISDR(clean, G(x))
     + lambda_anchor * L_logmag(T0(x), G(x))

L_E2 = L_E1 + lambda_p * L_PMSQE(clean, G(x))
```

`lambda_anchor` and `lambda_p` are fixed from train-only initial gradient
norms: the anchor contributes at most 50% and PMSQE at most 10% of the total
gradient norm at T0. They are recorded once and are not tuned on validation.

The frozen implementation contract is:

- `torch-pesq==0.1.2`, MIT, PyPI wheel
  `6f3fa836...`, whose core sources match upstream revision `3aac3c8...`;
- PMSQE frontend `512/256/512`, 49 Bark bands, factor `1`, WB/16 kHz only;
- MR-STFT Hann resolutions `256/64/256`, `512/128/512` and
  `1024/256/1024`, with equal magnitude/log-magnitude contribution;
- T0 anchor Hamming frontend `512/256/512` with `log1p(abs(STFT))`;
- true sample lengths for every loss; deterministic periodic extension only
  for PMSQE inputs shorter than its 20-frame minimum; silent clean targets
  contribute a neutral PMSQE term;
- gradient weights are measured on train-only, non-zero teacher-manifold
  perturbations around T0 because the exact T0 anchor gradient is zero.
- the direction support uses seed `3003`; weight calibration uses the first
  16 frozen direction-train identities, 32,000-sample segments and mask-logit
  delta `+0.02`, aggregating component waveform-gradient norms by their median.

The source/license/hash record is
`code_and_documentation/reference/torch_pesq_0.1.2.json`.

## Stage B — Local loss-direction audit

Create a new T3 support from `train_fit`, strictly pair/clean-utterance
disjoint from every T2 support identity:

| Partition | Utterances | Use |
|---|---:|---|
| direction train | 1,000 | candidate generation / D3 fitting |
| direction calibration | 200 | loss/critic selection and stopping |
| direction audit | 200 | one-shot local gate |

For every utterance, generate only teacher-manifold candidates:

- frozen T0 output;
- bounded perturbations of T0 mask logits;
- outputs from short train-only E1/E2 micro-trajectories;
- the current candidate after a proposed teacher update.

Do not use waveform interpolation as the primary local evidence. Compute true
PESQ-WB for every candidate and compare the predicted direction of E1 and E2
losses with the true PESQ delta around T0.

PMSQE may guide E2 only if the untouched direction audit has:

- at least 200 eligible local pairs;
- true/PMSQE improvement-sign agreement at least `70%`;
- delta Spearman at least `0.60`;
- no SNR quartile below `55%` sign agreement;
- finite, non-vanishing waveform and parameter gradients.

If this gate fails, E2 is not trained.

## Stage C — Matched E1/E2 teacher pilot

Start E1 and E2 from the same T0 hash, seed and batch order.

- learning rate `1e-6`;
- maximum 10 accepted epochs;
- ReduceLROnPlateau factor `0.5`, patience `2`, minimum `1e-7`;
- early stopping patience `3`;
- complete resumable model/optimizer/scheduler/patience/RNG history;
- evaluate `val_rank` after every proposed epoch;
- reject and roll back the entire epoch if PESQ-WB drops by more than `0.005`,
  STOI/SI-SDR guardrails are crossed, gradients become non-finite or the local
  perceptual-direction gate fails on the fixed calibration support.

After stopping, compare the selected E1 and E2 checkpoints once on
`val_select`. Test is not read.

E2 advances when it is safe, improves T0 by at least `+0.01` PESQ-WB and beats
E1. A safe positive result below `+0.01` is inconclusive, not promoted.

## Stage D — Conditional pairwise critic

Run only when E2 is blocked by its local gate or is safe but below the teacher
threshold. Do not run it after a harmful E2 result.

D3 keeps a scalar MetricGAN-style score but changes the training target from
absolute regression alone to local ranking:

```text
L_D3 = Huber(D(y), normalized_PESQ(y))
     + lambda_rank * softplus(
         -sign(delta_true) * (D(y_a) - D(y_b)) / temperature
       )
```

Pairs share the same noisy/clean utterance and come from actual T0/E1/E2
teacher-manifold candidates. Clean/noisy anchors remain auxiliary examples.
The fixed T3 direction audit is not used for optimization.

D3 must pass:

- global Spearman at least `0.80`;
- local sign agreement at least `75%`;
- local delta Spearman at least `0.65`;
- finite-difference gradient sign agreement at least `70%` on at least 200
  actual mask/parameter directions;
- no SNR quartile below `60%` local sign agreement;
- no range collapse or non-finite gradients.

Absolute MAE is reported but is not the primary gate because E3 uses the
pairwise score difference `D(G(x)) - D(T0(x))`.

If accepted, E3 uses the same schedule as E1/E2:

```text
L_E3 = L_E1
     + lambda_rank_g * softplus(
         -(D3(G(x), clean) - D3(T0(x), clean)) / temperature
       )
```

D3 is frozen during G updates. Its local gate is rechecked before every
accepted epoch on fixed calibration identities using current generator
outputs. The untouched direction audit is read only after D3 checkpoint
selection. A failure rolls back the epoch and stops E3.

## Stage E — Promotion and confirmation

Select at most one perceptual candidate, E2 or E3, on `val_select`.
Promotion requires:

- PESQ-WB gain over E0 at least `+0.01`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25 dB`;
- selected perceptual branch beats E1-SUP;
- three predeclared seeds reproduce a positive mean effect;
- paired utterance bootstrap 95% confidence interval for PESQ excludes zero;
- checkpoint/provenance/metrics pass independent audit.

Only then evaluate the selected teacher on test.

## Stage F — Student transfer

Only after Stage E:

1. create content-addressed local FP16 cache C3 from the accepted teacher hash;
2. train fresh S3-WB and S3-NB from zero;
3. keep S0 architecture, seeds, max-50, ReduceLR and early stopping;
4. evaluate WB with WB reference/PESQ-WB and NB with NB reference/PESQ-NB;
5. report `T3−T0`, `S3-WB−S0-WB` and `S3-NB−S0-NB`;
6. promote only audited code, metrics, figures and selected weights.

## Stop rules

- T0 identity/metric mismatch: repair before experiments.
- Direct-loss local gate fails: no E2 update.
- E1 and E2 both regress: stop; do not add D3 to a harmful base schedule.
- D3 local/finite-difference gate fails: no E3 update.
- Teacher gain below `+0.01`: no C3 or students.
- Any dataset mutation, test leakage, private path, CPU training or dirty
  snapshot invalidates promotion.

## Required outputs

- pinned PMSQE source/revision/license and parity fixtures;
- T3 support manifest/hash and T2-overlap report;
- local loss/critic direction tables and plots;
- E0/E1/E2/E3 matched histories and rollback evidence;
- true `val_rank`/`val_select` metrics with paired deltas;
- selected checkpoint and complete resume-state hashes;
- negative or positive article-ready audit;
- C3/S3 artifacts only if teacher promotion passes.

## Evidence basis

- MetricGAN+ connects a non-differentiable metric to enhancement through a
  learned discriminator: <https://arxiv.org/abs/2104.03538>
- MetricGAN+/- shows that predictor robustness depends on score support:
  <https://arxiv.org/abs/2203.12369>
- PMSQE defines a differentiable PESQ-inspired perceptual loss and warns
  against using its disturbance terms alone:
  <https://www.ugr.es/~joseangl/publication/martin-donas-deep-2018/martin-donas-deep-2018.pdf>
- A PyTorch PESQ-inspired implementation reports that combining the loss with
  SI-SDR is more stable than using it alone:
  <https://audiolabs.github.io/torch-pesq/>
- Pairwise/triplet ranking losses address limitations of independent score
  regression in speech-quality assessment:
  <https://www.isca-archive.org/interspeech_2024/ta24_interspeech.html>
