# Architecture register

Status: `clean-snapshot-pilot-confirmed`; no full run is promoted.

## A0 — End-to-end research pipeline

```text
[External read-only VoiceBank+DEMAND]
              |
              v
[Frozen manifests + split audit]
              |
              v
[MetricGAN+ WB FP32 teacher, 16 kHz]
      |                         |
      +--> [T0 vs T0_PESQ] --> [WB evaluation gate]
              |
              v
[Regenerable teacher cache]
      |                    |
      v                    v
[WB target, 16 kHz]   [NB target, 8 kHz]
      |                    |
      v                    v
[WB causal student]   [NB causal student]
      |                    |
      +---- D1 vs D1_PESQ -+
              |
              v
[profile gate -> optional QAT -> one final profile-matched test]
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
waveform
  -> STFT magnitude
  -> non-causal MetricGAN-like mask generator
     [2-layer bidirectional LSTM -> Linear -> Linear -> learnable sigmoid]
  -> magnitude mask + noisy phase
  -> iSTFT enhanced waveform
```

Canonical family alias: `metricgan_plus_teacher_wb`, variant `small`;
bandwidth `wb`; sample rate 16 kHz; frontend 512/160/320.

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
| `metricgan_plus_student_wb` | WB | 16 kHz | 96 | 1 | 128 | WB student |
| `metricgan_plus_student_nb` | NB | 8 kHz | 96 | 1 | 128 | NB student |

Both aliases use the same causal capacity so bandwidth is the controlled
variable. Historical `native8k_causal_*` family names remain readable only for
checkpoint/result compatibility and are not canonical experiment labels.

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
(source/noisy, candidate/enhanced, clean reference)
                    |
                    v
[frozen bandwidth-specific PESQ proxy]
                    |
                    v
[predicted PESQ] -> [- predicted PESQ] -> generator gradient
```

The non-differentiable PESQ implementation creates labels for a learned proxy;
PESQ itself is not differentiated through. `T0_PESQ` is the teacher metric
condition and `D1_PESQ`/`D2_PESQ` are student metric conditions. WB and NB
proxies have separate checkpoints and validation records.

`MetricGANGeneratorObjective` exposes the same optimization interface for a
future TTS generator. That extension is only `planned`: the enhancement proxy
must be recalibrated on outputs from the selected TTS system before use or
publication.

## A5 — Canonical campaign controller

```text
validate
  -> immutable manifest/profile checks
smoke-all
  -> T-WB anchor
  -> WB proxy labels/train/calibration
  -> T-WB-BASE + T-WB-METRIC
  -> val_select teacher choice
  -> one dual-profile teacher cache
  -> NB proxy labels/train/calibration
  -> S-WB-BASE + S-WB-METRIC
  -> S-NB-BASE + S-NB-METRIC
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
root.

Evidence:

- orchestration: `campaign.py`;
- proxy dataset/training/calibration:
  `code_and_documentation/sebench/metric_proxy_training.py`;
- teacher cache: `code_and_documentation/sebench/teacher_cache.py`;
- configuration: `configs/voicebank_campaign.yaml`;
- post-cleanup GPU smoke:
  `20260727-postcleanup-smoke-wbnb-s0-a5` (six cells, audit zero issues).
- clean-snapshot pilot:
  `20260727-pilot-wbnb-s0-a1` (six cells, 72 samples, audit zero issues).

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
