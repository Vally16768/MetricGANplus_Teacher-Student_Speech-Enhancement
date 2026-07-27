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
are keyed by teacher-checkpoint and training-manifest hashes, store teacher
waveforms and ERB masks in FP16, and leave noisy/clean cache fields empty so
the loader reads the external VoiceBank+DEMAND inputs.

Cause: caching avoids repeated teacher inference while preventing duplicated
dataset audio, Kingston writes and avoidable disk use. Cache precision is
validated on load; caches remain regenerable and are never Git artifacts.
