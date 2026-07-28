# Experiment register

## Lifecycle

```text
planned -> smoke-passed -> running -> evaluated -> audited -> promoted
                                      \-> failed/invalid/superseded
```

Only `promoted` runs belong in the canonical article-facing set.

Working runs are created under ignored `local/runs/<run_id>/`. After validation,
a sanitized, self-contained research record is promoted to
`experiments/runs/<run_id>/`; private path mapping stays outside Git.

Promotion requires:

- clean committed source snapshot;
- unique run ID and immutable run directory;
- resolved config, exact command, seed and environment;
- dataset/manifest hashes and split audit;
- teacher and student checkpoint ancestry;
- complete logs and raw metrics;
- evaluation sample counts;
- model/checkpoint and integrity hashes;
- plots and concise report;
- status `valid` from independent validation.

Failed, invalid and superseded runs must not remain as ambiguous model folders.
Keep only the lesson, failure cause and recoverability reference in an audit.
Delete or externally archive bulk artifacts only after their targets and hashes
are verified and the user explicitly authorizes removal.

## Canonical run layout

```text
experiments/runs/<run_id>/
  provenance/
    provenance.json
    config_resolved.yaml
    command.txt
    environment.txt
  logs/
  metrics/
    summary.json
    per_sample.csv
  models/
  reports/
  status.json
```

## Registry

| Run/set | Status | Evidence | Canonical |
|---|---|---|---|
| public reference checkpoints | observed | current repository | no |
| imported Kingston results | pending cleanup/audit | hash-verified copies | no |
| imported legacy worktree | historical dirty snapshot | hash-verified copies | no |
| `smoke_local_setup` | verification-only | `prepare_data` passed | no |
| `20260726-verification-smoke-wbnb-s0-a1` | failed smoke | CUDA device lacked explicit index; no dataset writes | no |
| `20260726-verification-smoke-wbnb-s0-a2` | superseded smoke | six cells passed, but reported audio samples were not retained | no |
| `20260726-verification-smoke-wbnb-s0-a3` | smoke-passed/audited | six cells, WB/NB proxies, dual cache, 36/36 samples, six model hashes, zero audit issues | no |
| `20260726-postcleanup-smoke-wbnb-s0-a4` | superseded smoke | six cells completed, but ERB extraction changed the working source while the process was active; proxy WB held-out correlation was negative on six smoke records | no |
| `20260727-postcleanup-smoke-wbnb-s0-a5` | smoke-passed/audited | stable post-cleanup source; six cells, six models, 36/36 reported samples, matched WB/NB protocols, zero audit issues | no |
| `20260727-pilot-wbnb-s0-a1` | pilot-passed/audited | clean commit `76729f3`; six cells, six models, 72/72 samples, frozen manifests unchanged, zero audit issues; verification-only | no |
| `20260727-full-wbnb-s0-a1` | stopped-by-user/invalid | clean commit `4fee1e3`; teacher/proxies/cache completed; stopped during epoch 14 validation of inadequate 96x1 `S-WB-BASE`; all 9.4 GiB preserved locally | no |
| official MetricGAN+ checkpoint diagnostic | observed/non-promotable | pinned revision and checkpoint hash; exact full test PESQ-WB 3.1225 on 824 pairs with the SpeechBrain adapter; exact local generator reconstruction passed four-row CUDA diagnostic | no |
| `20260727-official-two-stage-smoke-s0-a1` | superseded smoke | seven-cell graph passed, but failed gate used an ambiguous T1 downstream label | no |
| `20260727-official-two-stage-smoke-s0-a2` | superseded smoke | corrected T0 fallback and audited 7/7 cells, but identical stage-labeled caches were duplicated | no |
| `20260727-official-two-stage-smoke-s0-a3` | smoke-passed/audited | clean commit `8d36d62`; 7 cells/models, 42/42 samples, T0 fallback, one deduplicated FP16 cache, zero issues | no |
| `20260727-official-two-stage-pilot-s0-a1` | pilot-passed/audited; T1 gate failed | clean commit `0756a68`; 7 cells/models, 84/84 samples, strong fixed-proxy calibration but true-PESQ degradation; T0 fallback and cache reuse passed; zero audit issues | no |
| `20260727-bounded-teacher-smoke-s0-a1` | smoke-passed/audited | clean commit `27838d9`; bounded T1 stayed within 0.0007 PESQ of T0, failed the positive-gain gate, restored T0; 7 cells/models, 42/42 samples, zero issues | no |
| `20260727-bounded-teacher-pilot-s0-a1` | pilot-passed/audited; T1 gate failed | clean commit `33ef895`; stable bounded branches but best true PESQ remained epoch-0 T0; 7 cells/models, 84/84 samples, zero issues | no |
| `20260727-alternating-teacher-smoke-s0-a1` | smoke-passed/audited; superseded | clean commit `8df612f`; structural three-pass D/G execution exposed warm-start clean-label mismatch; preserved, no pilot use | no |
| `20260727-alternating-teacher-smoke-s0-a2` | smoke-passed/audited | clean commit `f5003ef`; corrected clean=1 target, current/history/current D refresh, local generated-only replay; 7 cells/models, 42/42 samples, zero issues; T0 fallback | no |
| `20260727-alternating-teacher-pilot-s0-a1` | execution-passed/audited; T1 gate failed | clean commit `9ad2b85`; 7 cells/models, 84/84 samples, zero issues; T1 val PESQ +0.00221 was below gate and test PESQ was -0.02029; T0 fallback | no |
| `20260727-official-baseline-smoke-s0-a1` | smoke-passed/audited | dirty-smoke implementation snapshot; exact three-cell T0→C0→S0 scope, no proxy/T1/S1, 3 models, 18/18 samples, zero cached inputs and zero audit issues | no |
| `20260727-official-baseline-full-s0-a1` | execution-passed/audited; students ceiling-limited | clean commit `357c1df`; 3/3 cells/models, 54/54 samples, zero issues; WB best epoch 20/20, NB best epoch 18/20; immutable max-50 continuation required | no |
| `20260727-student50-policy-smoke-s0-a1` | smoke-passed/audited | clean commit `330e501`; 3/3 cells/models, 18/18 samples, matched protocols and zero issues; verifies the max-50/scheduler/early-stop implementation path | no |
| `20260727-converged-s0-baseline-v1` | promoted then superseded | exact selected models retained; padded batched BLSTM evaluation superseded by v2 | no |
| `20260727-converged-s0-baseline-v2` | promoted/canonical | exact T0/S0 hashes, corrected true-length evaluation, artifact and privacy audits passed; commit `65b9a9c` | yes |
| `20260727-teacher-calibration-pilot-t1-a2` | execution-passed/audited; D gate failed | one strict refresh, 100 held-out, normalized MAE 0.1968, Pearson 0.5379, Spearman 0.5504; zero G updates | no |
| `20260727-teacher-calibration-retry-smoke-a1` | smoke-passed/audited | two-refresh flow, frozen-evaluation reuse, zero G updates and zero audit issues | no |
| `20260727-teacher-calibration-pilot-t1-a3` | execution-passed/audited; final D gate failed | two strict refreshes; final normalized MAE 0.2133, Pearson 0.5545, Spearman 0.5435; no E1/E2/C1/S1 | no |

The converged official-teacher S0 baseline v2 is the current promoted result.
There is no promoted T1/S1 result because the current-output calibration gate
failed before any generator update.

The smoke runs prove wiring only. The pilot used 256 training pairs and one
seed; it validates execution and exposes directional warnings, but its metric
values are not publication evidence and must not enter the article as final
results.

The first full attempt is also non-promotable. Its WB student improved from
1.837460 to a best 1.996045 PESQ-WB on `val_select`, but remained 0.218756
below the selected teacher on identical support. It was deliberately stopped
before the remaining student cells and replaced by a separately named
causal-max architecture. See
`docs/audits/2026-07-27-full-a1-stopped.md`.

### Pilot `20260727-pilot-wbnb-s0-a1`

The independent audit found six cells, six hashed model packages, 72 reported
audio files and zero issues. Profiles were matched: teacher/WB student used
16 kHz WB references and PESQ-WB; the NB student used 8 kHz NB references and
PESQ-NB. All four manifest hashes were unchanged and all cross-split overlaps
were zero.

Held-out proxy calibration was adequate for the engineering pilot:

| Proxy | Records train/validation | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| WB | 384 / 96 | 0.2488 | 0.9695 | 0.9022 |
| NB | 384 / 96 | 0.2437 | 0.9551 | 0.9503 |

True test-metric effects (`METRIC - BASE`, 64 pairs/profile):

| Pair | PESQ | STOI | SI-SDR | delta-SNR |
|---|---:|---:|---:|---:|
| teacher WB | -0.0107 | -0.00015 | -0.0309 | -0.0190 |
| student WB | +0.0344 | -0.00005 | -0.4050 | -0.4557 |
| student NB | +0.0306 | +0.00065 | +0.5494 | +0.4858 |

`T-WB-METRIC` was selected without test leakage because its val-select PESQ was
1.58617 versus 1.58553 for `T-WB-BASE`. The margin is only +0.00064 and its
test PESQ was lower by 0.01072. This is an explicit scientific warning, not a
reason to tune on the test split. The full, predeclared run is needed to
determine the teacher effect; no pilot result is promotable.

## Required reporting

Report PESQ, STOI, SI-SDR, delta-SNR and support counts for each split/domain.
Every metric row must include `bandwidth`, `sample_rate`, `pesq_mode` and
`reference_bandwidth`. Never average WB-PESQ and NB-PESQ into one number.
For declared multi-seed experiments report mean, standard deviation and 95%
confidence interval. Add model parameters, serialized size, latency protocol
and measured latency for deployment claims.

Graphs must include:

- train/validation loss;
- selection metric by epoch;
- teacher vs student vs QAT comparison;
- per-domain final metrics;
- latency/quality trade-off when deployment is claimed.

## Canonical ablation matrix

The replacement campaign is a controlled seven-cell, two-stage comparison.
Student architecture, split, seed and optimizer schedule stay fixed between S0
and S1; the teacher cache identity is the intended changed variable.

| Cell | Model | Band | Loss | Comparison |
|---|---|---|---|---|
| T0-WB-OFFICIAL | pinned official WB teacher | WB | epoch-0 checkpoint | credible teacher baseline |
| S0-WB | fresh causal-max WB student | WB | `D1` | distilled from T0 cache |
| S0-NB | fresh causal-max NB student | NB | `D1` | distilled from T0 cache |
| T1-WB-BASE | official teacher fine-tune control | WB | `T0` | training-control effect |
| T1-WB-METRIC | official teacher metric fine-tune | WB | `T0_PESQ` | PESQ-proxy effect |
| S1-WB | fresh causal-max WB student | WB | `D1` | distilled from gated T1 cache |
| S1-NB | fresh causal-max NB student | NB | `D1` | distilled from gated T1 cache |

Primary effect sizes are `T1 - T0`, `S1-WB - S0-WB` and
`S1-NB - S0-NB` for PESQ, STOI, SI-SDR and delta-SNR. The T1 teacher must gain
at least 0.01 PESQ-WB on `val_select`, lose no more than 0.002 STOI and lose no
more than 0.25 SI-SDR. Test metrics are reported only after that selection and
never choose the branch.

Required WB metric-proxy evidence:

- label-generation manifest and PESQ protocol;
- train/validation split by utterance identity;
- proxy MSE/MAE and rank correlation on held-out candidates;
- calibration plot and score-range coverage;
- frozen proxy checkpoint hash;
- generator ablation showing true-metric change, not only predicted change;
- explicit proxy-exploitation analysis when predicted and true PESQ disagree.

The direct student-metric experiment is deferred. If reintroduced, it requires
separate WB/NB proxies and a separately named matrix; it must not alter the
S1–S0 teacher-effect comparison.

## Student convergence policy

Canonical full S0 and S1 students use a 50-epoch ceiling with
validation-selected checkpoints, plateau LR reduction and eight-evaluation
early stopping. A run whose best checkpoint is its maximum epoch is
`ceiling-limited`: it may be structurally audited, but its student is not
treated as converged or promoted as the final student baseline.

`continue-students` creates a separate two-cell continuation package from an
audited official baseline. It preserves source state/model hashes and restores
the complete optimizer, scheduler, AMP scaler, history and best-score state.
WB and NB remain separate metric protocols throughout the continuation.

The audited full baseline confirms why this policy is required. `S0-WB`
selected epoch 20/20 with `val_select` PESQ-WB 2.596915. `S0-NB` selected
epoch 18 and stopped at 20 with `val_select` PESQ-NB 3.192184. The former is
unambiguously ceiling-limited, while the latter did not receive enough
post-best validation checks to satisfy the new early-stopping contract. See
`docs/audits/2026-07-27-official-baseline-full-a1.md`.

The TTS extension is a separate future campaign. It may reuse the objective
adapter but requires a selected TTS generator, its own outputs, a recalibrated
proxy and separate dataset/provenance. It cannot be promoted from the
VoiceBank enhancement results.

### Two-stage pilot `20260727-official-two-stage-pilot-s0-a1`

The official teacher achieved PESQ-WB 3.2626 and STOI 0.9266 on the 64-pair
pilot test support. The WB proxy had held-out Pearson 0.9539 and Spearman
0.9325, but the generator escaped that fixed calibration distribution:
`val_select` PESQ fell from 2.8238 to 2.4457 after one metric-aware epoch and
to 2.4285 after two. The control branch fell further. Both best checkpoints
therefore remained the epoch-0 official teacher.

The failed gate correctly reused the same content-addressed T0 cache for S0 and
S1. WB/NB student PESQ deltas were -0.00007/-0.00085, so no teacher-improvement
claim exists. The full campaign is blocked. See
`docs/audits/2026-07-27-two-stage-pilot-a1.md`.

### Bounded pilot `20260727-bounded-teacher-pilot-s0-a1`

The safety correction eliminated catastrophic proxy exploitation, but did not
create a positive teacher effect. Control `val_select` PESQ moved
2.8238 → 2.8093 → 2.7964; bounded metric PESQ moved
2.8238 → 2.8197 → 2.8131. Both selected the original checkpoint.

The frozen proxy remained strongly calibrated on fixed candidates (Pearson
0.9539), so repeating this full run would only scale a negative formulation.
The next experiment must implement current/noisy/historical discriminator
refresh as in MetricGAN+, then pass a new pilot gate.

### Alternating smoke `20260727-alternating-teacher-smoke-s0-a2`

The corrected implementation completed all seven cells and the independent
package audit with zero issues. D ran current/history/current updates and was
frozen for G. Its 344 KiB local replay stored only two FP16 enhanced outputs
and labels; noisy/clean inputs remained external.

The two-example D calibration was deliberately too small for inference
(Pearson -0.817). Both T1 branches restored T0 and the teacher gate failed.
This is a structural pass only; the monitored pilot is the next evidence gate.

### Alternating pilot `20260727-alternating-teacher-pilot-s0-a1`

The official T0 checkpoint produced test PESQ-WB 3.2626 and STOI 0.9266 on 64
pairs. The alternating branch moved true `val_select` PESQ from 2.8238 to
2.8226 and 2.8261. Its +0.00221 best gain missed the +0.01 gate; test PESQ was
0.02029 lower. T0 therefore remained downstream.

Current-output D calibration was inadequate despite a correlated warm start:
MAE degraded from 1.5002 to 1.7555 while generator-facing predicted PESQ moved
outside the warm-start prediction range. S1 reproduced S0 from the identical
T0 cache. Full remains blocked; the next trial is teacher-only and must repair
calibration before any S1 training. See
`docs/audits/2026-07-27-alternating-teacher-pilot-a1.md`.

### T2 discriminator successor — final negative outcome

`D2-OFFICIAL` restored exact SpeechBrain v1.1.0 frontend/model/update parity,
then fitted a fresh batch-size-one discriminator on a fixed
1,000/200/200 train/calibration/audit support. It failed both the scalar and
local-direction gates. The conditional `D2-RANGE` ablation added 7,000
train-only candidates with balanced PESQ-bin sampling while preserving the
same audit. It also failed and degraded local guidance.

| Strategy | Best/stop | nMAE | Pearson | Spearman | Local sign | Local rho |
|---|---:|---:|---:|---:|---:|---:|
| D2-OFFICIAL | 1 / 6 | 0.2895 | 0.7626 | 0.7768 | 0.5291 | -0.4929 |
| D2-RANGE | 6 / 11 | 0.3287 | 0.7283 | 0.7599 | 0.3266 | -0.6221 |

Neither checkpoint is approved for generator guidance. E1/E2, T2 teacher
confirmation, C2 and S2-WB/S2-NB are not applicable under the predeclared
gate. Sanitized negative packages, including their failed weights, remain in
`experiments/runs/20260728-t2-d2-*-negative/` for reproducibility and article
discussion.

### T3 direct-perceptual execution

| Run | State | Evidence | Promotable |
|---|---|---|---|
| `20260728-t3-direction-support-s0-a1` | direction gate passed | 5,600 candidates; audit sign `0.9222`, rho `0.8982`, min quartile `0.8454` | no, support only |
| `20260728-t3-e1-e2-smoke-a1` | CUDA flow passed | exact T0/cache/support hashes; E1/E2 update, rollback controls, WB evaluation and selection completed; E2 current-direction gate passed | no, two-file smoke |
| `20260728-t3-e1-e2-full-s3003-a1` | invalid infrastructure stop | missing pre-CUDA CuBLAS determinism contract; zero E1 optimizer steps; stopped before E2 | no |
| `20260728-t3-e1-e2-contract-smoke-a2` | corrected CUDA flow passed | clean planned contract adopted; deterministic E1/E2 updates and E2 current-direction recheck passed | no, two-file smoke |
| `20260728-t3-e1-e2-full-s3003-a2` | complete negative outcome | E1/E2 each rolled back at `1e-6`, `5e-7`, `2.5e-7`; selected checkpoints equal T0 and `val_select` gain is zero | no |

T3 and T4 are closed. T5 is the active teacher successor. Test, teacher cache
regeneration and students remain blocked.

### T4 bounded trust region

`20260728-t4-logit-bias-wb-s3003-a1` evaluated the exact ten-value scalar
mask-logit grid on `val_rank`, selected `-0.10`, then evaluated only that
checkpoint on `val_select`. PESQ-WB improved by `+0.002034`, while STOI and
SI-SDR changed by `-0.000467` and `-0.161412` dB. Both guardrails passed, but
the PESQ gain was below `+0.01`; the checkpoint is not promoted. This safe,
below-threshold outcome activates the predeclared T4-B micro-step
backtracking trial.

`20260728-t4b-microstep-wb-s3003-a1` completed exact-T0 trajectories at
1/4/16/64/256 steps and all 25 backtracking candidates. PESQ degraded with
horizon; the full 256-step proposal crossed the rollback boundary. The
selected H1/alpha-0.125 checkpoint changed `val_select` PESQ by `-0.000001`,
STOI by `+0.00000009` and SI-SDR by `+0.000074` dB. It is effectively T0 and
fails the teacher gate. T5 therefore replaced surrogate gradients with direct
true-PESQ search in a bounded smooth frequency curve.

`20260728-t5-frequency-wb-s3003-a1` used disjoint 96/96 train-only fit and
calibration supports. Its eight-knot sweep generalized positively through
`val_rank`; sweep 3 was selected. On `val_select`, PESQ improved `+0.005075`,
STOI changed `-0.000782` and SI-SDR `-0.245701` dB. The result is safe but
below the `+0.01` teacher gate and is retained as negative evidence.

`20260728-t6-affine-wb-s3003-a1` tested two retained T5 curves at seven global
logit scales on fresh disjoint 96/96 support. Scale `1.0` and T5 sweep 3 won,
so the `val_select` deltas were exactly the T5 result. Global temperature added
no improvement, the gate failed, and test remained unread. T7 is the active
confidence-conditioned successor.

`20260728-t7-confidence-wb-s3003-a1` tested 24 confidence-conditioned
low/high/threshold configurations on fresh disjoint 96/96 support. It selected
`low=-0.30/high=0/threshold=0/temperature=1.5`; `val_select` PESQ changed
`+0.004931`, STOI `-0.001245` and SI-SDR `-0.246202` dB. The gate failed and
test remained unread.

`20260728-t8-router-wb-s3003-a1` fitted a clean-free ridge router on 256
train-only examples and calibrated on 128 disjoint train examples. It selected
the T7 action for 75/128 examples and gained `+0.009197` PESQ with STOI
`-0.000957` and SI-SDR `-0.145724` dB. Its exact oracle ceiling was only
`+0.014197`, below the predeclared `+0.015` pre-validation gate, so validation
and test remained unread. T9 is the active four-action routing successor.

`20260728-t9-router-wb-s3003-a1` raised the four-action calibration oracle to
`+0.031116` PESQ. The PESQ-only router gained up to `+0.022907` but violated
both auxiliary guards. Its conservative threshold `0.02` retained `+0.017368`
PESQ and passed STOI (`-0.001773`), while SI-SDR (`-0.279120` dB) missed the
unchanged limit by `0.029120` dB. It stopped before validation and test. T10
is the active fresh-support conservative-margin successor.

`20260728-t10-router-wb-s3003-a1` selected margin `0.025`. It gained
`+0.008331` on `val_rank` and `+0.008015` PESQ on `val_select`, with STOI
`-0.001639` and SI-SDR `-0.207664` dB. Both auxiliary guards passed, but the
unchanged PESQ gate failed; test remained unread. T11 is the active
risk-penalized action-selection successor.
