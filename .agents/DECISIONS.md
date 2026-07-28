# Decision log

## D-001 — One canonical MetricGAN+ repository

Decision: use this repository for MetricGAN+ teacher–student only.

Cause: MP-SENet and MetricGAN+ results were mixed, making provenance and article
claims ambiguous.

## D-002 — Dataset remains external and unchanged

Decision: read frozen data/manifests externally; write all experiment outputs
outside dataset roots.

Cause: protect the dataset and make code/results independently manageable.

## D-003 — Use normal Git, not Git LFS

Decision: keep selected checkpoints in normal Git.

Cause: individual files are below the hosting per-file limit and the user
prefers a simpler repository.

## D-004 — Commit/push authorized after release gate

Decision: commit and push are authorized from 2026-07-26 onward, but only after
tests and the project guard pass.

Cause: versioned snapshots are now needed to track changes and unlock
promotable experiments. Machine-specific legacy material remains a release
blocker until it is removed from the current public tree or safely sanitized.

## D-005 — Canonical set contains only valid end-to-end runs

Decision: promote only audited, complete runs; keep failed/superseded bulk out of
the canonical set.

Cause: duplicated and partial artifacts create contradictory interpretations and
make article writing unreliable.

## D-006 — One WB teacher and two bandwidth-specific students

Decision: train one WB/16 kHz MetricGAN+ teacher, then maintain equal-capacity
WB/16 kHz and NB/8 kHz causal students as separate experiment tracks.

Cause: bandwidth must be an explicit experimental variable, not an accidental
property inferred from mixed result folders.

## D-007 — VoiceBank+DEMAND is the only canonical training dataset

Decision: exclude DNS5 and combined-dataset strategies from all new canonical
training. Preserve them as historical evidence until cleanup is authorized.

Cause: isolate the scientific question and keep the dataset unchanged.

## D-008 — Metrics and references follow the model bandwidth

Decision: WB runs use WB references and PESQ-WB; NB runs use NB references and
PESQ-NB. Other intrusive metrics use the same aligned, profile-specific pair.

Cause: a score is only interpretable under the correct signal bandwidth and
reference protocol.

## D-009 — Metric-discriminator study is an ablation

Decision: compare baseline versus PESQ-proxy loss for teacher, WB student and NB
student. Maintain separate WB/NB proxy checkpoints and validate improvements
with true metrics.

Cause: prevent proxy exploitation and attribute any change to the metric-aware
objective.

## D-010 — TTS transfer is separate and initially planned

Decision: add a reusable differentiable generator-objective adapter, but do not
claim TTS improvement until a proxy is recalibrated on a selected TTS
generator's outputs and evaluated in a separate campaign.

Cause: an enhancement-trained proxy is out of domain for synthesized speech.

## D-011 — Shared environment and GPU-only training

Decision: every training/cache command must run from the shared project virtual
environment and on CUDA. CPU is limited to preparation, tests and audits.

Cause: eliminate environment drift and accidental CPU training.

## D-012 — Remove legacy public surface

Decision: remove machine runbooks, DNS/combined configurations, alternative
model families and obsolete orchestration from the current tree. Preserve
recovery through public commit `5129bae` and ignored local forensic imports.

Cause: the user explicitly authorized cleanup so the canonical MetricGAN+
campaign can pass privacy/scope gates and produce a clean training snapshot.

## D-013 — Replace canonical students with recovered causal-max capacity

Decision: use new canonical aliases
`metricgan_plus_student_wb_causal_max` and
`metricgan_plus_student_nb_causal_max`, both with a three-layer
unidirectional GRU (`hidden_size=160`) and a 224-unit linear projection.
Retain the former 96x1 aliases only for historical checkpoint compatibility.

Cause: the first full WB student showed measurable but inadequate learning,
remaining 0.218756 PESQ-WB below the selected teacher on the same validation
support. The stronger architecture is the exact MetricGAN-style student used
for WB/NB in the earlier MP-SENet teacher–student campaign.

Constraint: transfer architecture only. Do not warm-start from the historical
weights because their provenance includes non-VoiceBank training data.

## D-014 — Anchor the campaign to the official MetricGAN+ teacher

Decision: stage T0 uses the pinned official
`speechbrain/metricgan-plus-voicebank` WB generator without retraining. Two
fresh causal-max students, WB and NB, are trained from its content-addressed
cache. Stage T1 fine-tunes control and PESQ-proxy teacher branches from the
same official checkpoint, promotes a branch only after a true-PESQ
`val_select` gain with STOI/SI-SDR guardrails, regenerates a separate cache and
trains two fresh S1 students with the same seeds and schedules as S0.

Cause: the former teacher was a simplified reimplementation trained for only
10 epochs with a mismatched frontend and objective; its test PESQ-WB of 2.529
was not comparable to the official MetricGAN+ result above 3. The two-stage
design establishes a credible baseline first and then attributes S1–S0 changes
only to the teacher upgrade.

## D-015 — Persist only regenerable teacher targets in local FP16 caches

Decision: teacher caches live only in the ignored Desktop-local runtime area,
are keyed by teacher-checkpoint, training-manifest and cache-contract hashes,
store teacher waveforms and ERB masks in FP16, and leave noisy/clean cache
fields empty so the loader reads the external VoiceBank+DEMAND inputs. Stage
labels are metadata and cannot create duplicate content for the same identity.

Cause: caching avoids repeated teacher inference while preventing duplicated
dataset audio, Kingston writes and avoidable disk use. Cache precision is
validated on load; caches remain regenerable and are never Git artifacts.

## D-016 — Bound teacher metric optimization and anchor it to T0

Decision: normalize predicted PESQ from `[-0.5, 4.5]` to `[0, 1]` and minimize
MSE to the MetricGAN clean target score `1`. During T1 fine-tuning, use the
content-addressed T0 cache as a waveform trust anchor with weight `0.75`, use
the official 512/256/512 Hamming/log-spectral feature loss and lower the
fine-tune learning rate to `1e-5`.

Cause: pilot `20260727-official-two-stage-pilot-s0-a1` showed Pearson 0.9539 on
fixed held-out proxy candidates, while unconstrained generator updates
increased predicted training PESQ and reduced true `val_select` PESQ from
2.8238 to 2.4457 after one epoch. The original SpeechBrain recipe minimizes
score error toward a bounded normalized target and retrains its discriminator;
it does not maximize raw predicted PESQ.

Constraint: the current frozen-proxy branch is a safety-corrected ablation, not
an exact reproduction of the official alternating discriminator/history loop.
Full training remains blocked unless true PESQ passes the predeclared gate.

## D-017 — Stop scaling the frozen proxy; implement discriminator refresh

Decision: do not run the bounded frozen-proxy T1 at full scale. The next teacher
experiment must alternate discriminator and generator updates and refresh the
discriminator with true normalized PESQ labels for clean, noisy, current
enhanced and historical enhanced examples.

Cause: bounded pilot A1 eliminated collapse but still reduced true
`val_select` PESQ from 2.8238 to 2.8197 after one metric epoch and 2.8131 after
two. Fixed-candidate proxy Pearson remained 0.9539, demonstrating that static
calibration is not enough for generator-induced distribution shift. The
original SpeechBrain MetricGAN+ recipe explicitly retrains the discriminator
before each generator epoch and replays historical outputs.

Constraint: test metrics cannot tune this loop. Selection remains confined to
the frozen `val_select` split and the existing promotion guardrails.

## D-018 — Use the SpeechBrain discriminator and local generated-only replay

Decision: canonical T1 uses the SpeechBrain MetricGAN discriminator architecture
(four 5x5 spectral-normalized convolutions, mean pooling and 50/10/1 linear
head). Before every generator epoch it executes current clean/enhanced/noisy,
historical enhanced and current clean/enhanced/noisy D updates. Clean uses the
exact target `1`; noisy/enhanced use `(PESQ + 0.5) / 5`. D is then frozen for
the generator update.

Generated current enhanced waveforms are cached as FP16 only inside the ignored
Desktop run directory. Replay metadata references VoiceBank noisy/clean files
in place and caches noisy PESQ scores, but never copies dataset inputs.

Cause: this is the defining behavior and architecture of the original
SpeechBrain recipe missing from both failed frozen-proxy pilots. It directly
addresses generator-induced distribution shift while preserving the external
read-only dataset and reproducible resume state.

Constraint: implementation status is not evidence of improvement. C30 passed
its structural smoke, but the following pilot failed; full remains gated by
true `val_select` PESQ and STOI/SI-SDR.

## D-019 — Stop downstream work after a failed teacher gate

Decision: the next fidelity experiment is teacher-only. It must use at least
100 current outputs per discriminator refresh, record a current-output
calibration guard, and stop before regenerating teacher caches or training S1
students unless T1 gains at least 0.01 true PESQ-WB on `val_select` while
passing the STOI/SI-SDR guardrails.

Cause: alternating pilot `20260727-alternating-teacher-pilot-s0-a1` produced
only +0.00221 `val_select` PESQ and -0.02029 test PESQ. Current-output D MAE
degraded from 1.5002 to 1.7555 and predicted PESQ exceeded the warm-start
calibration range. The T0 anchor prevented collapse, but repeated S1 training
after the failed gate only reproduced S0 and consumed GPU time.

Constraint: test remains reporting-only. Do not tune discriminator epochs,
learning rates or stopping from the observed test delta.

## D-020 — Materialize the official baseline before changing the teacher

Decision: execute the research program as three separately auditable phases.
Phase 1 contains only the pinned official T0 teacher, one content-addressed
Desktop-local WB/NB FP16 cache and fresh S0-WB/S0-NB students. Phase 2 changes
only the WB enhancement teacher and stops at its true-metric gate. Phase 3
creates C1 and fresh S1 students only after T1 passes.

Cause: the official teacher is already credible and must establish the student
baseline independently. Mixing T1 development into the same launch wastes GPU
time after failed gates and makes it harder to distinguish baseline evidence
from teacher-improvement evidence.

Constraint: the TTS idea is a separate domain experiment. It may reuse the
metric-generator interface, but a TTS generator requires its own data,
calibration and evaluation; its output cannot directly supervise the
VoiceBank enhancement students.

## D-021 — Use validation-governed 50-epoch student training

Decision: full WB and NB student runs use a maximum of 50 epochs,
`ReduceLROnPlateau` with factor 0.5/patience 2/minimum LR `1e-6`, and early
stopping after eight non-improving bandwidth-matched `val_select` evaluations.
Ceiling-limited historical students are continued only into a new immutable run
with complete optimizer/scheduler/scaler/history restoration.

Cause: full S0-WB reached its highest PESQ-WB at the predeclared epoch-20
ceiling after the LR had only recently fallen to `2.5e-4`. The run therefore
established a valid executed checkpoint but did not establish convergence.

Constraint: test remains reporting-only. Continuations must retain the source
manifest/cache hashes and cannot overwrite or relabel the epoch-20 package.

## D-022 — Use a calibration-gated teacher-improvement protocol

Decision: after the S0 baseline is closed, resume robustness is verified and
the sanitized baseline is promoted, the next T1 work follows
`.agents/TEACHER_IMPROVEMENT_PLAN.md`. It first calibrates the
SpeechBrain discriminator on at least 100 held-out current-generator outputs
without updating G, then compares the frozen official T0, a conservative
fine-tuning control and a metric-aware branch. A generator update is permitted
only while the current-output calibration gate passes.

Cause: the alternating pilot gained only `+0.00221` true `val_select`
PESQ-WB, while current-output discriminator MAE degraded and its generator
scores escaped the calibrated range. The next experiment must distinguish
ordinary fine-tuning from a genuine metric-discriminator effect and prevent an
unreliable proxy from steering the generator.

Constraint: the plan does not alter completed S0 evidence. T1 still requires
at least `+0.01` PESQ-WB with STOI/SI-SDR guardrails before C1 or S1 work.
The synthesis hypothesis remains a separate TTS campaign with its own
generator outputs, targets, calibration, data, evaluation and claims.

## D-023 — Permit one fixed-generator discriminator recalibration retry

Decision: the calibration-only diagnostic may execute exactly two
current/history/current refreshes. The same strict held-out gate applies after
the second refresh, and failure stops T1 before E1/E2. Generator evaluation is
reused while G is frozen; no epoch or final reevaluation is repeated between
calibration refreshes.

Cause: strict A2 used 100 update plus 100 disjoint held-out current outputs
and failed safely with normalized MAE 0.1968, Pearson 0.5379 and Spearman
0.5504. The predeclared plan allowed one refresh retry, while the observed
three identical full E0 evaluations added cost without scientific evidence.

Constraint: the retry does not relax thresholds, change the dataset, use test
results, update G or authorize a broader hyperparameter search.

## D-024 — Replace T1 with a separately gated T2 successor

Decision: preserve T1 as final negative evidence and execute any further
teacher work under the T2/D2 namespace defined by
`.agents/TEACHER_SUCCESSOR_PLAN.md`. T2 first establishes numerical parity
with the official SpeechBrain discriminator recipe, then fits a fresh
batch-1 D to convergence on disjoint T0-output train/calibration/audit
partitions. A teacher update requires both the existing scalar calibration
gate and a new local directional gate around T0.

Cause: the final permitted T1 retry reduced the discriminator's internal
update loss but failed held-out fidelity with normalized MAE `0.2133`,
Pearson `0.5545`, Spearman `0.5435` and out-of-range predictions. The official
recipe trains D with batch size one, repeated current/history/current passes
and a long alternating schedule; T1's padded-batch warm start and two refreshes
did not establish a reliable current-output surrogate. A scalar predictor can
also have acceptable global error while providing a harmful local generator
gradient, so local PESQ-delta direction must be tested explicitly.

Constraint: the D thresholds are not relaxed. Score-support widening is one
conditional, predeclared train-only ablation and cannot use test data. No G,
C2, S2-WB or S2-NB work begins after a failed parity or D2 gate. TTS remains a
separate campaign.
