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
| [EXECUTION_TODO.md](EXECUTION_TODO.md) | Completed P1–P6 execution board, dependencies and evidence | historical baseline/T1 audit |
| [TEACHER_IMPROVEMENT_PLAN.md](TEACHER_IMPROVEMENT_PLAN.md) | Executed calibration-first T1 protocol and separate future TTS-transfer boundary | T1 audit or future successor design |
| [TEACHER_SUCCESSOR_PLAN.md](TEACHER_SUCCESSOR_PLAN.md) | Predeclared T2 method for discriminator fidelity, controlled teacher improvement and gated student transfer | all T2 design or implementation work |
| [TEACHER_SUCCESSOR_TODO.md](TEACHER_SUCCESSOR_TODO.md) | Completed T2 dependency board and negative D2 evidence | T2 audit or successor diagnosis |
| [TEACHER_T3_PLAN.md](TEACHER_T3_PLAN.md) | Predeclared direct-perceptual and conditional pairwise-critic teacher successor | all T3 design or implementation work |
| [TEACHER_T3_TODO.md](TEACHER_T3_TODO.md) | Active T3 dependency board, gates, evidence and next action | start and end of every T3 iteration |

Operational skill:
[`skills/manage-metricgan-research/SKILL.md`](skills/manage-metricgan-research/SKILL.md).

Current release state: **the corrected true-length S0 baseline v2 is promoted
and canonical. S0-WB selected epoch 34/stop 42 and S0-NB selected 41/49.
T1 failed calibration. T2 then restored exact official parity, but both
D2-OFFICIAL and D2-RANGE failed scalar/local-direction gates, so there is no
T1/T2 teacher, C1/C2 or S1/S2 result. T3 is planned but not implemented**.

Open work:

- T3.1 must pin and validate the direct differentiable perceptual loss;
- no E2/E3 teacher update may start before its local-direction gate passes;
- the TTS metric-critic hypothesis remains a separate future campaign.
