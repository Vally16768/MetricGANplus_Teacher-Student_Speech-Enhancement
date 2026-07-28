# Architecture register

Status: `teacher-successor-t3-pilot`; converged S0 is published, E0 remains
frozen, and the matched direct-perceptual pilot is active.

## A0 — End-to-end research pipeline

```text
[External read-only VoiceBank+DEMAND]
              |
              v
[Frozen manifests + split audit]
              |
              v
[Official MetricGAN+ WB checkpoint T0]
              |
              +--> [local cache C0: WB + NB targets]
              |             |
              |             +--> [fresh WB student S0]
              |             +--> [fresh NB student S0]
              |
              +--> [D2 official-parity calibration]
                            |
                            v
                   [D2 scalar + local-direction gates]
                            |
                            v
              +--> [T2 control + T2 MetricGAN fine-tuning]
                            |
                            v
                  [true WB val_select gate]
                            |
                            +--> [local cache C1: WB + NB targets]
                                          |
                                          +--> [fresh WB student S1]
                                          +--> [fresh NB student S1]
                                                        |
                                                        v
                                      [profile-matched test + paired deltas]
              |
              v
[Metrics + curves + model + provenance + report]
```

Cause: isolate one academically traceable MetricGAN+ teacher–student line and
prevent MP-SENet or alternative-teacher results from contaminating claims.

Evidence:

- pipeline controller: `campaign.py`;
- portable I/O/profile contracts:
  `code_and_documentation/sebench/contracts.py`;
- model builders: `code_and_documentation/sebench/models.py`;
- training loop: `code_and_documentation/sebench/training.py`;
- bandwidth contract: `code_and_documentation/sebench/bandwidth.py`;
- bandwidth-matched ERB frontend shared by losses and teacher-cache generation:
  `code_and_documentation/sebench/erb.py`;
- canonical campaign contract:
  `code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml`;
- plan validator: `code_and_documentation/sebench/research_plan.py`;
- portable campaign config: `configs/voicebank_campaign.yaml`;
- local-only manifest binder: `scripts/bind_voicebank_manifests.py`.

## A1 — Teacher

```text
waveform at 16 kHz
  -> STFT [FFT 512, hop 256, window 512, Hamming]
  -> log1p(magnitude)
  -> official non-causal mask generator
     [2-layer BLSTM, hidden 200/direction
      -> Linear 400x300 + LeakyReLU
      -> Linear 300x257
      -> 257-bin learnable sigmoid]
  -> expm1(mask * log-magnitude) + noisy phase
  -> iSTFT enhanced waveform
```

Canonical family alias: `metricgan_plus_teacher_official_wb`, variant `small`;
bandwidth `wb`; sample rate 16 kHz; frontend 512/256/512; 1,895,514 trainable
parameters. Initialization is the pinned official
`speechbrain/metricgan-plus-voicebank` generator at revision
`a196ce26b3bdace6fa1d819017584bdbcce462a8`, checkpoint SHA-256
`147bfb866bac8264603546e035bf283370e716ed2f4b7412d308d2bcee88304f`.
All 21 generator tensors must load and none may be skipped.

Saved repository checkpoint packages contain the complete generator state and
set `initialize_from_official=false`, so evaluation, caching and fine-tuning
resume offline without an implicit model download.

Historical MetricGAN checkpoint family names remain loadable for compatibility,
but are not canonical public experiment names. FullSubNet, MP-SENet, CMGAN,
STM32 simulation and machine-specific teacher strategies were removed from the
current tree. The ERB operations still required by MetricGAN+ distillation now
live in the model-neutral `sebench/erb.py` module.

## A2 — Student

```text
waveform
  -> padded STFT magnitude^(1/2)
  -> unidirectional GRU
  -> Linear + LeakyReLU
  -> Linear + learnable sigmoid mask
  -> masked magnitude + noisy phase
  -> iSTFT waveform
```

| Canonical family | Band | Rate | Hidden | GRU layers | Linear | Role |
|---|---:|---:|---:|---:|---:|---|
| `metricgan_plus_student_wb_causal_max` | WB | 16 kHz | 160 | 3 | 224 | WB student |
| `metricgan_plus_student_nb_causal_max` | NB | 8 kHz | 160 | 3 | 224 | NB student |

Both aliases use the same causal capacity so bandwidth is the controlled
variable. The recurrent graph is frame-causal; centered STFT analysis imposes
a fixed 16 ms lookahead for both profiles. The WB and NB models contain
604,386 and 514,018 trainable parameters respectively because their spectral
input/output dimensions differ.

The recovered design is the `causal_max` student used in the historical
MP-SENet teacher–student campaign. It is a MetricGAN-style magnitude-mask
student, not the full MP-SENet magnitude/phase teacher architecture. Only its
architecture is transferred; its mixed-dataset weights are excluded.

The old `metricgan_plus_student_wb` and `metricgan_plus_student_nb` aliases
remain loadable at 96 hidden units/one GRU layer solely so pilot and stopped-run
checkpoints do not change meaning. They are no longer canonical campaign
families. Historical `native8k_causal_*` family names likewise remain readable
only for checkpoint/result compatibility.

QAT uses fake quantization in the causal mask generator. The final deployment
claim requires a validated exported model and measured latency/model size, not
only the presence of QAT code.

Canonical full student training has a 50-epoch ceiling, selects checkpoints
only by bandwidth-matched `val_select/PESQ`, and uses
`ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-6)`. Early
stopping waits eight non-improving validation epochs, leaving a recovery window
after an LR reduction. If an earlier immutable run ended at its epoch ceiling,
`continue-students` restores its model, optimizer, scheduler, AMP scaler,
history, selection state, RNG state and train-loader generator state into a new
run directory; it never overwrites the source package. After every evaluation,
the durable state is written only after the plateau scheduler, best checkpoint,
patience counter and history row are updated. An evaluation state resumes at
the next epoch rather than repeating the completed epoch.

## A3 — Bandwidth and metric protocol

| Model profile | Model input | Clean reference | PESQ mode | STOI/SI-SDR/delta-SNR |
|---|---|---|---|---|
| teacher WB | 16 kHz WB | the paired clean file loaded at 16 kHz | `wb` | same 16 kHz aligned pair |
| student WB | 16 kHz WB | the paired clean file loaded at 16 kHz | `wb` | same 16 kHz aligned pair |
| student NB | 8 kHz NB | the same paired clean identity loaded at 8 kHz | `nb` | same 8 kHz aligned pair |

`evaluate_manifest` emits `bandwidth`, `reference_bandwidth`, `sample_rate` and
`pesq_mode`. A mismatch stops evaluation. WB and NB PESQ values are reported in
separate columns/figures and are not pooled.

Evaluation always invokes the enhancer separately at each utterance's true
length. This is required for the bidirectional teacher: right-padding is
future context to a BLSTM and previously made E0 metrics depend on
`eval_batch_size`. The requested evaluation batch now controls iteration/
loading only, not the waveform passed to the model. A regression test verifies
that two unequal utterance lengths reach the model unpadded.

## A4 — Metric discriminator and generator objective

```text
before each G epoch:
 current clean ------------target 1-----------------> D updates
 current enhanced/noisy ---true (PESQ+0.5)/5-------> D updates
 historical enhanced --------stored true score------> D replay
 current clean/enhanced/noisy ----------------------> D updates
                                      |
                                      v
                freeze SpeechBrain D [4x Conv2D spectral norm
                                      -> channel mean -> 50 -> 10 -> 1]
                                      |
 current enhanced + clean ------------+--> MSE(D score, 1)
                                              + T0 trust anchor
                                              -> generator gradient
```

The non-differentiable PESQ implementation creates labels for D; PESQ itself is
not differentiated through. `T0_PESQ` is the stage-T1 teacher metric condition.
D matches SpeechBrain's four valid-convolution, spectral-normalized
architecture and its current/history/current refresh order. T2 parity also
uses the exact SpeechBrain discriminator frontend:
`log1p(abs(STFT))`, constant centered padding and
`[batch, time, frequency]` layout. The earlier T1 frontend incorrectly used
`sqrt(abs(STFT))`, reflect padding and a transposed layout; all T1
discriminator evidence remains historical and is not reused as D2 evidence.
The pinned parity source is SpeechBrain `v1.1.0` revision
`36c180c7bfad3bf5c48bd76a24799812952c4565`, recorded in
`code_and_documentation/reference/speechbrain_metricgan_v1.1.0.json`.
The generator target is the official normalized clean score `1`; D is frozen
during G.
T1 also reads the local T0 teacher cache and anchors its waveform to the
accepted official output; the fine-tune learning rate is `1e-5`.

Generated current teacher outputs and historical replay live only inside the
ignored Desktop run directory as FP16. Their index stores true enhanced/noisy
PESQ labels (clean is fixed to 1) and references the external clean/noisy paths without copying
dataset audio. Noisy scores are reused from a local JSON cache. D optimizer,
checkpoint and refresh history are part of the resumable T1 state.

T2 reuses the complete content-addressed official T0 WB cache instead of
regenerating identical outputs. `metricgan_d2.py` selects fixed
train/calibration/audit identities, computes true PESQ-WB for noisy/T0
candidates, estimates input SNR, verifies source hashes before/after and writes
only local records/plots. It does not copy noisy or clean audio. Because source
speaker/noise labels were not retained by content-addressed staging, the
support is provably pair/clean-utterance disjoint but not claimed
speaker-disjoint.

`fit_d2_official` performs two clean/enhanced/noisy current passes with one
enhanced historical pass between them, all at batch 1. It selects D only on
the fixed calibration partition using normalized MAE with correlation/range
penalties, persists model/optimizer/plateau-scheduler/patience/history after
every evaluation, and reads the audit partition only after selection. The
final audit combines the strict scalar fidelity gate with true-PESQ local
direction tests for bounded T0→clean and T0→noisy interpolations.

The active T1 fidelity protocol draws two disjoint current-output partitions
per refresh: at least 100 examples for D updates and at least 100 held out for
calibration. The held-out partition reports raw/normalized MAE and RMSE,
Pearson, Spearman, target/prediction ranges and prediction variance. A failed
gate persists the D/replay evidence but skips the generator epoch, scheduler
decision and selection update. Resume binds the D model, D optimizer, refresh
history and exact replay root alongside the existing G/optimizer/scheduler/
patience/RNG/history state.

The canonical S0/S1 student comparison uses `D1` in both stages, with identical
architecture, seed and schedule, so the changed teacher is the only intended
experimental variable. A future direct student-metric ablation must restore
distinct WB/NB proxies and cannot be mixed into this teacher-effect experiment.

T3 is a separate, direct-perceptual teacher successor and does not reuse either
failed T2 discriminator:

```text
VoiceBank noisy WB/16 kHz --------------------------> official T0 mask model
             |                                                |
             |                                                +--> frozen T0 output
             |                                                |
             +--> same initialized trainable model -----------+--> candidate
                                                                  |
clean WB/16 kHz --> MR-STFT + 0.10 SI-SDR ------------------------+
frozen T0 output --> Hamming 512/256/512 logmag trust anchor -----+
clean WB/16 kHz --> pinned torch-pesq 0.1.2 surrogate (E2 only) --+
                                                                  |
                                                calibrated E1/E2 total loss
                                                                  |
                              true PESQ-WB local-direction gate --+--> update or stop
```

`t3_perceptual.py` enforces true sample lengths, WB/16 kHz PMSQE input,
multi-resolution `256/64/256`, `512/128/512`, `1024/256/1024` Hann STFTs,
and the official T0 Hamming anchor frontend. The external surrogate is
PESQ-inspired, not exact ITU PESQ, and is pinned with source/license hashes in
`code_and_documentation/reference/torch_pesq_0.1.2.json`. Its weight and the
anchor weight are frozen from train-only local gradient norms; validation does
not tune them.

The official teacher exposes bounded `+/-0.10` mask-logit candidate generation
for the T3 local-direction audit. A zero perturbation is bit-identical to the
ordinary forward path, the parameter/state-dict architecture is unchanged,
and these candidates remain ignored FP16 local artifacts.
`t3_support.py` freezes 1,000/200/200 train-only identities with both pair and
clean-utterance exclusion against all supplied T2 supports. The root campaign
entry point owns selection, independent reconciliation and 16-row CUDA
gradient-weight calibration; neither validation nor test is read.
It then creates four bounded mask-logit variants (`-0.04`, `-0.02`, `+0.02`,
`+0.04`) per identity, stores only generated FP16 waveforms, labels them with
true PESQ-WB and the direct surrogate, and applies the untouched audit gate.

`t3_training.py` owns the isolated matched E1/E2 trainer. Both branches load
the same official T0 hash and complete FP16 T0 cache, use Adam at `1e-6`,
batch-size-one deterministic 32,000-sample segments and identical
seed/order/crops. The cache dataset supplies the true unpadded segment length.
Every proposed epoch is accepted only after true `val_rank` WB metrics and,
for E2, a current-output local PMSQE/PESQ direction recheck. Rejection restores
the complete pre-epoch model, optimizer, scheduler and RNG state before
reducing LR. Durable state exists only at post-evaluation boundaries and
contains the selected state, counters, history and source hashes.
The root `train-t3-teacher --resume` command binds the original clean commit
and support contract, reuses completed E0/branch summaries, and resumes an
incomplete branch only from that atomic post-evaluation state.
For a new production run it also adopts the skill-generated planned run
contract only when its config hash and clean Git commit match exactly.

T4-A keeps the network graph unchanged and folds one bounded uniform
mask-logit shift into the 257-bin `linear2.bias`. It scans a frozen grid with
true PESQ-WB on `val_rank`, rejects STOI/SI-SDR violations, and reads
`val_select` only for the single selected delta. The ordinary checkpoint is
therefore offline-loadable without a runtime wrapper.

T4-B restarts each declared micro-step horizon from exact T0. Its
PMSQE-primary objective retains the frozen PMSQE coefficient and scales the
MR-STFT/SI-SDR/T0-anchor constraint block by `0.10`. Each horizon produces an
ordinary checkpoint; interpolation back toward T0 at the frozen alpha grid
implements the trust-region line search. True WB `val_rank` selects at most
one candidate and only that candidate reads `val_select`.

T5 keeps the neural graph frozen and folds a smooth 257-bin mask-logit bias
curve into the ordinary output-layer bias. Eight bounded frequency knots are
optimized by deterministic true-PESQ coordinate search on a train-only fit
support. A disjoint train-only calibration support is read only after complete
sweeps; `val_rank` selects among T0, the uniform T4-A start and completed
sweeps; only one candidate reads `val_select`.

T6 expands only the final-layer calibration to
`scale * original_logit + frequency_curve`. It folds the scale into the
257-output linear weight/bias and adds one of two frozen T5 curves. A fresh
train-only 96/96 fit/calibration support filters a fixed seven-scale grid
before `val_rank` and one-shot `val_select`.

T7 adds a checkpoint-configured, confidence-conditioned transform after the
official final linear layer and before its learnable sigmoid:
`z + low + (high-low)*sigmoid((z-threshold)/temperature)`. It uses only the
teacher's own logits at inference. A fixed 24-candidate grid is filtered on
fresh disjoint train-only 96/96 fit/calibration support, then at most two
candidates reach `val_rank` selection and one reaches `val_select`. Disabled
calibration is exactly the original T0 path; checkpoint reconstruction restores
the transform from portable `model_config`.

T8 wraps exact T0/T7 behavior in one checkpoint-configured utterance router.
It computes a frozen 16-value feature vector from the noisy waveform, official
log-magnitude frontend, logits, masks and bounded T7 correction. A standardized
ridge score selects either the exact base mask or exact T7 mask per utterance.
Ridge labels and threshold selection use only fresh train identities; clean
audio is not part of the inference graph. Oracle and learned-router
pre-validation gates can stop the experiment before `val_rank`.

The earlier bounded frozen-proxy branch remains historical negative evidence,
not the canonical T1 implementation. The alternating branch passed structural
smoke but failed its clean pilot promotion gate: current-output D calibration
degraded and the true `val_select` PESQ gain was only +0.00221. The next
experiment must remain teacher-only until calibration and the true-metric gate
pass; code presence and structural execution do not establish improvement.

`MetricGANGeneratorObjective` exposes the same optimization interface for a
future TTS generator. That extension is only `planned`: the enhancement proxy
must be recalibrated on outputs from the selected TTS system before use or
publication.

## A5 — Canonical campaign controller

```text
validate
  -> immutable manifest/profile checks
prepare-d2-support
  -> bind canonical T0 cache hash
  -> fixed 1000/200/200 train/calibration/audit identities
  -> true PESQ-WB labels + coverage/source-mutation audit
prepare-d2-range-support
  -> preserve the fixed D2 calibration/audit identities
  -> derive train-only noisy/T0/clean interpolations and bounded output masks
  -> cache only derived FP16 candidates outside Git/dataset
  -> balance fitting candidates across raw PESQ-WB bins
smoke-d2[-range] / train-d2[-range]
  -> batch-1 current/history/current D fitting
  -> calibration-only checkpoint selection and stopping
  -> one-shot fixed scalar and local-directional audit
smoke-baseline / pilot-baseline / run-baseline
  -> T0-WB-OFFICIAL at epoch 0
  -> persistent dual-profile cache C0
  -> S0-WB + S0-NB from fresh identical schedules
  -> three-cell true-metric report + independent audit
  -> stop before proxy/T1/S1
smoke-all
  -> T0-WB-OFFICIAL at epoch 0
  -> persistent dual-profile cache C0
  -> S0-WB + S0-NB from fresh identical schedules
  -> WB proxy labels/train/calibration
  -> T1-WB-BASE + T1-WB-METRIC from the official checkpoint
  -> true-metric val_select gate against T0
  -> persistent dual-profile cache C1
  -> S1-WB + S1-NB from fresh identical schedules
  -> true-metric aggregation + plots + model hashes + report
pilot-all
  -> same graph on a larger frozen subset; clean source required
run-all
  -> same graph with full manifests/hyperparameters; clean source required
continue-students
  -> immutable epoch-20 S0 states
  -> restore model/optimizer/scheduler/scaler/history
  -> max-50 plateau-LR + early stopping
close-baseline
  -> independently audit epoch-20 baseline + continuation
  -> bind T0 from baseline and selected S0-WB/S0-NB from continuation
  -> recompute 20-to-converged deltas
  -> convergence plot + final tables + model/source hashes
  -> three-cell converged-baseline package + independent audit
smoke-resume
  -> one uninterrupted CUDA control
  -> one fault-injected stop after a durable evaluation checkpoint
  -> resume to the same final epoch
  -> compare LR, scheduler, patience, best/final model and history state
smoke-teacher-calibration / calibrate-teacher
  -> frozen promoted E0 teacher from canonical true-length S0 v2
  -> disjoint current D-update and held-out calibration supports
  -> at most two predeclared current/history/current refreshes
  -> reuse frozen E0 metrics between refreshes; no generator update
  -> explicit calibration gate + predicted-versus-true plot
smoke-teacher / pilot-teacher
  -> require an audited, passed calibration-only source
  -> E0 frozen + E1 control + E2 metric branch only
  -> skip every E2 G epoch whose current-output calibration fails
  -> true WB val_select teacher gate; never build C1 or train S1 here
smoke-t3-teacher / train-t3-teacher
  -> require the passed T3 untouched direction audit
  -> evaluate immutable T0 once on val_rank and val_select
  -> matched deterministic E1-SUP and E2-PMSQE from the same T0/cache/seed
  -> per-proposal rollback, plateau LR, early stopping and exact resume state
  -> E2 current-output local direction recheck on frozen calibration identities
  -> true WB val_select gate; never read test or train students
scan-t4-logit-bias
  -> adopt clean planned contract and exact failed-T3 baseline
  -> true-WB val_rank scan of bounded uniform mask-logit bias
  -> one selected val_select evaluation and unchanged teacher gate
  -> never read test or train students
smoke-t4-microstep / train-t4-microstep
  -> exact T0 restart at each bounded train-step horizon
  -> PMSQE-primary update with supervised/anchor constraints
  -> checkpoint interpolation and true-WB val_rank backtracking
  -> one selected val_select evaluation; never test/cache/students
smoke-t5-frequency / search-t5-frequency
  -> freeze disjoint train-only fit/calibration support
  -> optimize eight bounded frequency knots with true PESQ, no surrogate
  -> hard STOI/SI-SDR constraints at fit/cal/rank/select
  -> ordinary selected checkpoint; never test/cache/students
smoke-t6-affine / search-t6-affine
  -> fresh disjoint 96/96 train-only support
  -> two frozen curves x seven exact final-logit scales
  -> top-5 calibration, top-3 rank, one selected val_select
  -> ordinary checkpoint; never test/cache/students
smoke-t7-confidence / search-t7-confidence
  -> fresh disjoint 96/96 train-only support
  -> fixed confidence-conditioned low/high/threshold grid
  -> top-8 fit, top-4 calibration, top-2 rank, one selected val_select
  -> checkpoint round-trip; never test/cache/students
smoke-t8-router / search-t8-router
  -> fresh train-only 256/128 fit/calibration support
  -> true-PESQ T7-minus-T0 labels and fixed 16-feature ridge router
  -> oracle/generalization/auxiliary gates before val_rank
  -> one frozen rank and conditional val_select; never test/cache/students
promote-baseline
  -> accept only an audited/promotable converged S0 closure
  -> preserve corrective true-length evaluation provenance when present
  -> copy selected T0/S0-WB/S0-NB weights with SHA-256 verification
  -> retain aggregate metrics, student histories, plots and report
  -> replace machine paths with portable environment bindings
  -> exclude dataset/cache/audio/replay/training-state artifacts
  -> canonical run-contract + independent package/privacy audit
monitor-run / audit-run
  -> live stage/cell state / independent package reconciliation
```

`smoke-all` is allowed on dirty source only with the explicit
`--allow-dirty-smoke` flag; the same applies to `smoke-baseline`.
`pilot-all`, `run-all`, `pilot-baseline` and `run-baseline` require a clean
source snapshot. Smoke and pilot are marked `verification_only` and cannot be
promoted. A clean full baseline may be promoted independently of the later T1
gate. All training nodes enforce the shared venv and CUDA contract.
Generated files live below the configured run root, never below the dataset
root. Teacher caches are content-addressed by teacher checkpoint, training
manifest and cache-contract hashes, remain in the Desktop-local ignored runtime
area, store teacher waveforms/masks as FP16 and do not duplicate noisy/clean
dataset audio. Stage labels do not duplicate identical cache content.

Evidence:

- orchestration: `campaign.py`;
- proxy dataset/training/calibration:
  `code_and_documentation/sebench/metric_proxy_training.py`;
- alternating current/history/current update and local replay:
  `code_and_documentation/sebench/metricgan_alternating.py`;
- teacher cache: `code_and_documentation/sebench/teacher_cache.py`;
- matched T3 pilot: `code_and_documentation/sebench/t3_training.py`;
- T4 bounded calibration: `code_and_documentation/sebench/t4_calibration.py`;
- T4 micro-step backtracking:
  `code_and_documentation/sebench/t4_microstep.py`;
- T5 zeroth-order frequency curve:
  `code_and_documentation/sebench/t5_zeroth_order.py`;
- T6 affine-logit search: `code_and_documentation/sebench/t6_affine.py`;
- T7 confidence-conditioned search:
  `code_and_documentation/sebench/t7_confidence.py`;
- T8 train-only adaptive router:
  `code_and_documentation/sebench/t8_router.py`;
- T9 train-only multi-action adaptive router:
  `code_and_documentation/sebench/t9_multi_router.py`;
- T10 conservative-risk margin calibration:
  `code_and_documentation/sebench/t10_risk_router.py`;
- configuration: `configs/voicebank_campaign.yaml`;
- post-cleanup GPU smoke:
  `20260727-postcleanup-smoke-wbnb-s0-a5` (six cells, audit zero issues).
- clean-snapshot pilot:
  `20260727-pilot-wbnb-s0-a1` (six cells, 72 samples, audit zero issues).
- first full attempt:
  `20260727-full-wbnb-s0-a1` (user-stopped during the inadequate 96x1 WB
  student; preserved and non-promotable).
- official-checkpoint diagnostic: 21/21 tensors loaded, 1,895,514 parameters,
  PESQ-WB 3.3407 on the four-row frozen smoke test support.
- final two-stage smoke:
  `20260727-official-two-stage-smoke-s0-a3` (seven cells/models, 42 samples,
  failed-gate T0 fallback, one deduplicated cache, audit zero issues).
- monitored two-stage pilot:
  `20260727-official-two-stage-pilot-s0-a1` (seven cells/models, 84 samples,
  strong fixed-proxy calibration but true-PESQ degradation, T0 fallback, audit
  zero issues). This blocks full training until the teacher objective is made
  bounded and robust to current-output distribution shift.
- bounded-objective smoke:
  `20260727-bounded-teacher-smoke-s0-a1` on commit `27838d9` (seven
  cells/models, 42 samples, no T1 collapse, T0 fallback, audit zero issues).
- bounded-objective pilot:
  `20260727-bounded-teacher-pilot-s0-a1` on commit `33ef895` (seven
  cells/models, 84 samples, stable but negative T1 teacher changes, T0
  fallback, audit zero issues).
- corrected alternating-D smoke:
  `20260727-alternating-teacher-smoke-s0-a2` on commit `f5003ef` (seven
  cells/models, current/history/current refresh and generated-only local
  replay, 42 samples, T0 fallback, audit zero issues).
- alternating-D pilot:
  `20260727-alternating-teacher-pilot-s0-a1` on commit `9ad2b85` (seven
  cells/models, 84 samples, +0.00221 `val_select` PESQ below gate,
  current-output D calibration degradation, T0 fallback, audit zero issues).
- official-baseline-only smoke:
  `20260727-official-baseline-smoke-s0-a1` (three expected/observed cells,
  official cache reuse, no proxy/T1/S1, 18 sample files and zero audit issues).
- official student continuation:
  `20260727-official-students-cont50-s0-a1` (WB best 34/stop 42, NB best
  41/stop 49, both early-stopped; two cells/models and zero audit issues).
- converged-baseline closure is generated only through `close-baseline`, which
  verifies the baseline/continuation ancestry and source model hashes before
  writing the merged report.
- resume-equivalence validation uses `smoke-resume`; its fault injection occurs
  only after the post-evaluation state is atomically persisted.
- CUDA resume-equivalence smoke A4 passed with identical LR, patience, best
  state, selected-model hash and history after a planned interruption. It runs
  real CUDA forward/backward while freezing optimizer effect for the exact
  equivalence comparison because the canonical reflection-pad backward kernel
  does not provide deterministic CUDA gradients.

## A6 — Selection boundaries

- `train_fit`: optimization only.
- `val_rank`: frequent within-run ranking.
- `val_select`: candidate/model selection.
- `test`: final hold-out reporting; no hyperparameter decisions.

Any architecture change must record:

1. changed block and cause;
2. source/config changes;
3. parameter and compute impact;
4. compatibility with historical checkpoints;
5. required tests and reruns;
6. new source hashes in `.agents/state/architecture_sources.sha256`.
