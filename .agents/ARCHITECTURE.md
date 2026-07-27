# Architecture register

Status: `alternating-discriminator-implemented-awaiting-gpu-smoke`; no full run
is promoted.

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
              +--> [T1 control + T1 alternating MetricGAN fine-tuning]
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

## A3 — Bandwidth and metric protocol

| Model profile | Model input | Clean reference | PESQ mode | STOI/SI-SDR/delta-SNR |
|---|---|---|---|---|
| teacher WB | 16 kHz WB | the paired clean file loaded at 16 kHz | `wb` | same 16 kHz aligned pair |
| student WB | 16 kHz WB | the paired clean file loaded at 16 kHz | `wb` | same 16 kHz aligned pair |
| student NB | 8 kHz NB | the same paired clean identity loaded at 8 kHz | `nb` | same 8 kHz aligned pair |

`evaluate_manifest` emits `bandwidth`, `reference_bandwidth`, `sample_rate` and
`pesq_mode`. A mismatch stops evaluation. WB and NB PESQ values are reported in
separate columns/figures and are not pooled.

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
D now matches SpeechBrain's four valid-convolution, spectral-normalized
architecture and its current/history/current refresh order. The generator
target is the official normalized clean score `1`; D is frozen during G.
T1 also reads the local T0 teacher cache and anchors its waveform to the
accepted official output; the fine-tune learning rate is `1e-5`.

Generated current teacher outputs and historical replay live only inside the
ignored Desktop run directory as FP16. Their index stores true enhanced/noisy
PESQ labels (clean is fixed to 1) and references the external clean/noisy paths without copying
dataset audio. Noisy scores are reused from a local JSON cache. D optimizer,
checkpoint and refresh history are part of the resumable T1 state.

The canonical S0/S1 student comparison uses `D1` in both stages, with identical
architecture, seed and schedule, so the changed teacher is the only intended
experimental variable. A future direct student-metric ablation must restore
distinct WB/NB proxies and cannot be mixed into this teacher-effect experiment.

The earlier bounded frozen-proxy branch remains historical negative evidence,
not the canonical T1 implementation. The alternating branch still requires a
clean real-GPU smoke and pilot; code presence alone does not establish teacher
improvement.

`MetricGANGeneratorObjective` exposes the same optimization interface for a
future TTS generator. That extension is only `planned`: the enhancement proxy
must be recalibrated on outputs from the selected TTS system before use or
publication.

## A5 — Canonical campaign controller

```text
validate
  -> immutable manifest/profile checks
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
monitor-run / audit-run
  -> live stage/cell state / independent package reconciliation
```

`smoke-all` is allowed on dirty source only with the explicit
`--allow-dirty-smoke` flag. `pilot-all` and `run-all` require a clean source
snapshot. Smoke and pilot are marked `verification_only` and cannot be
promoted. All training nodes enforce the shared venv and CUDA contract.
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
