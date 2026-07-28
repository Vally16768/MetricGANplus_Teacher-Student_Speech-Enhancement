---
name: manage-metricgan-research
description: Govern and iteratively execute the MetricGAN+ teacher–student speech-enhancement repository as a clean, reproducible academic project. Use for project discovery, progress checks, TODO tracking, architecture or pipeline changes, data/split configuration, code edits, tests, audits, experiment planning or training, metric/plot/model management, cleanup, documentation, provenance validation, result promotion, or article preparation in this repository.
---

# Manage MetricGAN+ Research

Treat `.agents/` as the project control plane and the dataset as an external
read-only input.

## Start every task

1. Locate the repository root.
2. Read `AGENTS.md` and `.agents/INDEX.md`.
3. Read the active execution board named by `.agents/INDEX.md`
   (`.agents/TEACHER_T11_TODO.md` for T11), then only the other
   task-specific registers linked by the index. Read
   `.agents/EXECUTION_TODO.md` when completed P1–P6/T1 provenance is needed.
4. Inspect Git state, affected files and existing evidence without cleaning or
   normalizing the worktree.
5. State the intended block, experiment or document and its cause.
6. Reconcile the single active item in the active execution board with
   processes and immutable evidence. Update it after every material iteration.
7. Synchronize campaign-wide status in `.agents/TODO.md`. Never mark a gate
   passed from plans or code presence alone; attach the required evidence.

Read [references/contracts.md](references/contracts.md) for artifact, run and
evidence schemas. Read [references/checklists.md](references/checklists.md) for
the workflow matching the task.

## Enforce scope

- Keep MetricGAN+ teacher–student only.
- Use only VoiceBank+DEMAND for canonical training and keep it read-only.
- Train the teacher as WB/16 kHz. Maintain exactly two student tracks:
  WB/16 kHz and NB/8 kHz.
- Bind every objective metric to the model profile: WB reference and PESQ-WB
  for WB; NB reference and PESQ-NB for NB. Never compare scores produced under
  different bandwidth protocols as if they were one metric.
- Use a WB metric-discriminator checkpoint for canonical T1 teacher
  fine-tuning. Before every T1 generator epoch, update the SpeechBrain
  four-convolution discriminator in current clean/enhanced/noisy, historical
  enhanced, current clean/enhanced/noisy order with clean target `1` and true
  normalized PESQ labels for noisy/enhanced. Freeze D during the generator
  update. If a direct student-metric
  ablation is separately declared, use distinct WB and NB proxies; a WB proxy
  is not valid for NB, and an enhancement proxy is not TTS evidence.
- Reject MP-SENet, FullSubNet, CMGAN and unrelated project artifacts from the
  canonical pipeline.
- Keep datasets, audio, generated caches and machine-local configs out of Git.
- Reject personal paths, usernames, credentials, host/IP details, mount names,
  server-specific scripts and external-drive orchestration from publishable
  files.
- Preserve historical evidence until cleanup is verified; never silently edit
  it to appear portable.

Run
`.agents/skills/manage-metricgan-research/scripts/project_guard.py --repo <repo>`
before proposing publication, commit or push. Treat every error as a blocker,
not as a cosmetic warning.

## Execute the iterative board

- Treat `.agents/TEACHER_T11_TODO.md` as the detailed source of truth for the
  active T11 sequence. Treat `.agents/TEACHER_T10_TODO.md`,
  `.agents/TEACHER_T9_TODO.md`,
  `.agents/TEACHER_T8_TODO.md`,
  `.agents/TEACHER_T7_TODO.md`,
  `.agents/TEACHER_T6_TODO.md`,
  `.agents/TEACHER_T5_TODO.md`,
  `.agents/TEACHER_T4_TODO.md`,
  `.agents/TEACHER_T3_TODO.md` and
  `.agents/TEACHER_SUCCESSOR_TODO.md` as completed negative evidence and
  `.agents/EXECUTION_TODO.md` as the completed P1–P6/T1 evidence ledger;
  `.agents/TODO.md` remains the campaign summary.
- Keep at most one subtask `in-progress` and name one concrete `Next action`.
- Execute the first unblocked dependency only. Do not start T1 before the S0
  baseline is audited, resume robustness passes and the baseline is promoted.
- At each progress check, inspect the process and run artifacts before editing
  status. Absence of a process is not proof of successful completion.
- Record run ID, evidence path or artifact hash for every passed gate.
- When a task fails, preserve its cause and set downstream tasks to `blocked`;
  never silently advance the next phase.
- End each material iteration by updating the board header and progress log,
  even when the result is a controlled failure or no state change.

## Change code or configuration

1. Identify the affected architecture block and evidence source.
2. Record why the change is needed and which claims/runs it invalidates.
3. Make the smallest coherent change.
4. Add or update proportionate tests.
5. Run unit tests.
6. Run the real repository-root `campaign.py` entry point when imports, paths,
   model, data, training or evaluation behavior can change.
7. Update `.agents/ARCHITECTURE.md` and
   `.agents/state/architecture_sources.sha256` when architecture sources change.
8. Update validation and documentation registers in the same change.

Do not launch training from dirty or uncommitted code.

All training commands are GPU-only and must run from the shared project virtual
environment identified by `METRICGAN_SHARED_VENV` or the repository-sibling
`shared-venv`. CPU remains allowed for read-only preparation, tests and audits.

For T3, treat `torch-pesq` strictly as a pinned PESQ-inspired surrogate. Require
its source/license record, WB/16 kHz contract, finite CPU/CUDA gradients and the
untouched true-PESQ local-direction gate before any E2 optimizer step. Freeze
anchor/PMSQE weights from train-only non-zero teacher-manifold directions;
never tune those weights on `val_rank`, `val_select` or test. A failed local
gate blocks E2 but preserves its evidence and activates only the conditional
D3 path declared by the T3 plan.

## Plan and run experiments

Use `scripts/run_contract.py create` only after tests pass and the code snapshot
is clean. It creates the private working run under ignored `local/runs/`. Give
each run a unique immutable ID.

Require:

- exact commit and clean-worktree status;
- resolved config, command, seed and environment;
- manifest hashes and split validation;
- initialization checkpoint ancestry;
- new output directory;
- logs, raw metrics, support, plots and models.

Execute in order:

```text
phase 1: prepare_data -> split audit -> pinned official WB teacher T0
-> content-addressed local WB+NB cache C0 -> fresh S0-WB + S0-NB
-> baseline report + independent audit
phase 2: T1 control/metric teacher fine-tuning -> true-metric teacher gate
phase 3: content-addressed local WB+NB cache C1 -> fresh S1-WB + S1-NB
-> paired bandwidth-matched evaluation -> report -> audit
```

Use `smoke-baseline`, `pilot-baseline` or `run-baseline` for phase 1. These
commands must stop after the three-cell T0/S0 package and must not build a
proxy, alter the teacher or train S1. Audit the expected cell set declared by
the package rather than assuming every run is a seven-cell comparison.

Stop downstream work when a gate fails. Do not reinterpret a partial run as
end-to-end evidence.

The canonical teacher gate requires a predeclared positive true PESQ-WB delta
on `val_select` plus STOI/SI-SDR guardrails. Test is never a selection input.
Smoke/pilot may continue after a failed gate only as explicitly
`verification_only`; such runs can never be promoted.

After one verification run has demonstrated failed-gate fallback and cache
identity, isolate later T1 iterations as teacher-only trials. Use at least the
original recipe's 100 current examples per discriminator refresh, audit
calibration on current generator outputs and stop before cache generation or
S1 training unless the teacher passes its true-metric gate. Do not repeat
students merely to reconfirm an unchanged T0 fallback.

Teacher caches must stay in the ignored Desktop-local runtime area, outside
the dataset and Git. Key them by teacher checkpoint, frozen training manifest
and cache contract; stage labels must not duplicate identical content. Store
regenerable teacher targets in validated FP16, and do not copy noisy/clean
dataset audio into the cache.

Metric-discriminator replay follows the same location rule. Cache only
generated enhanced outputs as FP16 plus their PESQ labels and external input
references. Reuse noisy scores locally, never copy noisy/clean audio, and
record current-output calibration for every discriminator refresh.

Use `campaign.py monitor-run --run-dir <run-dir>` throughout pilot/full
execution. Reconcile the active campaign stage with the current cell epoch,
metrics and generated files before allowing the next gate.

For multi-stage campaign requests, maintain these gates in the active execution
board named by `.agents/INDEX.md` and synchronize the summary in
`.agents/TODO.md`:

```text
audit -> local manifest binding -> canonical flow implementation
-> unit/integration tests -> real GPU smoke -> monitored pilot -> full run
-> independent collection/report -> promotion audit -> cleanup audit
```

Dataset path repair is allowed only by generating ignored local manifest copies.
Record source and resolved hashes, prefix mapping and missing-row count. Never
rewrite a source manifest in place.

For the TTS research extension, reuse only the differentiable generator
objective interface. Recalibrate and validate the metric discriminator on
outputs from the selected TTS generator before any synthesis claim. Keep that
domain's dataset, configs, results and claims separate from the VoiceBank
speech-enhancement campaign.

## Validate and promote results

Use `scripts/run_contract.py validate --stage canonical`.
Use `campaign.py audit-run --run-dir <run-dir>` first to reconcile the declared
three-cell baseline or seven-cell comparison CSV, profile metadata, applicable
baseline/teacher gate, reported samples, model sizes/hashes and report
artifacts.

Promote only when:

- the real pipeline completed;
- metrics independently reconcile with raw outputs;
- sample counts and split identities match;
- checkpoint ancestry is known;
- required graphs/models exist and hashes match;
- every reported sample/artifact path exists in the run package;
- status is `valid`;
- claims use the evidence language in `.agents/ACADEMIC.md`.

Keep only promoted runs in the canonical experiment set. For failed, invalid or
superseded runs, retain a concise cause/lesson and recoverability reference.
Remove bulk only with explicit authorization after verifying hashes and targets.
Before promotion, replace machine-local paths with stable logical dataset and
artifact identities while retaining private exact provenance outside Git.
Load repository-produced PyTorch checkpoints through restricted
`weights_only=True` deserialization unless a separately audited legacy
conversion explicitly requires otherwise.

## Maintain architecture and documentation

Express architecture as compact blocks with:

- stable block ID;
- inputs/outputs;
- components and dimensions;
- causal/non-causal behavior;
- source/config evidence;
- status and last validation.

Update the architecture register and source hashes in the same change.

Maintain every canonical Markdown file in
`.agents/DOCUMENTATION_INDEX.md`. Consolidate duplicated facts into one source
of truth. Mark historical claims explicitly; do not carry stale values into the
article.

## Prepare academic outputs

Build tables and figures only from promoted run records. Trace every claim from
article text to aggregate metrics, raw predictions, checkpoint, config, commit
and manifest hashes. Report seeds, mean, standard deviation, 95% confidence
interval, support and per-domain results where applicable.

Never commit or push without explicit user authorization.
