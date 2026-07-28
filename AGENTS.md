# Project instructions

Use the project-local skill at
`.agents/skills/manage-metricgan-research/SKILL.md` for every change to code,
data configuration, experiments, metrics, models, reports, audits, architecture,
or publication material.

Before acting:

1. Read `.agents/INDEX.md`.
2. Read `.agents/TEACHER_T3_TODO.md` and reconcile its active item with
   evidence. Use `.agents/TEACHER_SUCCESSOR_TODO.md` for completed T2 evidence
   and `.agents/EXECUTION_TODO.md` for completed P1–P6/T1 provenance.
3. Read the task-specific canonical registers linked from the index.
4. Inspect the worktree without cleaning, resetting, deleting, or normalizing it.
5. Keep MetricGAN+ teacher–student in scope; MP-SENet is a separate project.

Non-negotiable rules:

- Treat datasets as external read-only inputs.
- Use only VoiceBank+DEMAND for new canonical training.
- Train one WB/16 kHz teacher and keep WB/16 kHz and NB/8 kHz students as
  separate experiment lines.
- Evaluate each model against its matching-band clean reference and PESQ mode.
- Run training only from the shared project venv and only on CUDA.
- Never write outputs, caches, tracking, or manifests under a dataset root.
- Never overwrite a historical run or silently edit a historical artifact.
- Do not publish personal paths, usernames, credentials, hostnames, IP addresses,
  mount names, external-drive logic, or server-specific orchestration.
- Every code change requires proportionate tests and the real entry-point smoke
  test when runtime paths can be affected.
- Update `.agents/ARCHITECTURE.md` and its source-hash baseline whenever model,
  training, pipeline, or canonical-config behavior changes.
- Update `.agents/DOCUMENTATION_INDEX.md` whenever Markdown documentation is
  added, removed, renamed, or changes status.
- Only promoted, valid end-to-end runs belong in the canonical experiment set.
  Failed or superseded runs may leave a concise audit lesson, not ambiguous
  models and duplicated outputs.
- Keep at most one detailed execution item `in-progress`; update its evidence,
  next action and progress log after every material iteration.
- Do not commit or push without explicit user authorization.
