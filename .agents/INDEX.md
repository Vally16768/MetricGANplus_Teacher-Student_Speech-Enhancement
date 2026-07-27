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

Operational skill:
[`skills/manage-metricgan-research/SKILL.md`](skills/manage-metricgan-research/SKILL.md).

Current release state: **the official T0 teacher is pinned and validated; an
independent three-cell T0→C0→S0 baseline phase is being prepared before any
new teacher modification. Earlier T1 formulations failed the PESQ gate**.

Open blockers:

- Historical experiment validity and checkpoint ancestry are not yet fully
  reconstructed.
- The old six-cell pilot remains non-promotable and does not validate the new
  official-teacher/two-stage graph.
- The pinned official teacher, local reconstruction and full seven-cell graph
  pass structure, checkpoint, CUDA smoke and independent package audit gates.
- The clean pilot calibrated the frozen WB proxy well, but both T1 branches
  reduced true `val_select` PESQ and were restored to the official checkpoint.
- The bounded target-score/T0-anchor pilot was stable but still negative; the
  next branch must refresh its discriminator on current/noisy/historical
  outputs as in MetricGAN+.
- There is no promoted end-to-end run produced from the current clean snapshot.
- The official full baseline must complete before the next teacher-only metric
  trial and before any S1 retraining.
