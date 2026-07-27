# Project control index

This directory is the canonical control plane for the research project.

| Register | Purpose | Read when |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current system blocks, interfaces, evidence and source map | model, pipeline or config work |
| [DATA.md](DATA.md) | Dataset contract, split policy and read-only boundary | any data or training work |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Run lifecycle, registry and promotion state | experiments, metrics, models |
| [VALIDATION.md](VALIDATION.md) | Test matrix and required gates | every code/config change |
| [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) | Status and ownership of every canonical document | documentation or article work |
| [ACADEMIC.md](ACADEMIC.md) | Evidence language, reporting and article-ready outputs | claims, reports, publication |
| [DECISIONS.md](DECISIONS.md) | Decisions and their causes | before changing project policy |
| [TODO.md](TODO.md) | Ordered cleanup/implementation/experiment gates | every active campaign task |
| [EXECUTION_TODO.md](EXECUTION_TODO.md) | Detailed P1–P6 execution board, dependencies, evidence and next action | at the start and end of every project iteration |
| [TEACHER_IMPROVEMENT_PLAN.md](TEACHER_IMPROVEMENT_PLAN.md) | Deferred calibration-first T1 plan and separate TTS-transfer protocol | after S0 completion, before C32/C37 implementation |

Operational skill:
[`skills/manage-metricgan-research/SKILL.md`](skills/manage-metricgan-research/SKILL.md).

Current release state: **the official T0 teacher is pinned; the max-50 S0-WB
and S0-NB continuation completed through early stopping and its two-cell
package audit has zero issues. The merged epoch-20→converged baseline report,
resume repair and promotion remain ahead of any new teacher modification**.

Open blockers:

- The original epoch-20 baseline and immutable continuation still need one
  merged report and independent closure audit.
- The post-evaluation scheduler/early-stopping resume state needs a deterministic
  interrupt/resume repair and CUDA smoke before the next experiment.
- There is no promoted, sanitized S0 baseline package yet.
- Teacher-only T1 remains blocked until S0 promotion; S1 remains blocked until
  the true-metric teacher gate passes.
