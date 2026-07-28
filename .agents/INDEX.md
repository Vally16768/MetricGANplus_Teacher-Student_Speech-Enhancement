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
| [TEACHER_T3_TODO.md](TEACHER_T3_TODO.md) | Completed T3 dependency board and negative E1/E2 evidence | T3 audit or successor diagnosis |
| [TEACHER_T4_PLAN.md](TEACHER_T4_PLAN.md) | Executed bounded true-PESQ trust-region successor | T4 audit or successor diagnosis |
| [TEACHER_T4_TODO.md](TEACHER_T4_TODO.md) | Completed T4 board and negative scalar/micro-step evidence | T4 audit |
| [TEACHER_T5_PLAN.md](TEACHER_T5_PLAN.md) | Executed true-PESQ frequency-curve successor | T5 audit |
| [TEACHER_T5_TODO.md](TEACHER_T5_TODO.md) | Completed T5 board and below-gate evidence | T5 audit |
| [TEACHER_T6_PLAN.md](TEACHER_T6_PLAN.md) | Executed affine-logit successor | T6 audit |
| [TEACHER_T6_TODO.md](TEACHER_T6_TODO.md) | Completed T6 board and below-gate evidence | T6 audit |
| [TEACHER_T7_PLAN.md](TEACHER_T7_PLAN.md) | Executed confidence-conditioned successor | T7 audit |
| [TEACHER_T7_TODO.md](TEACHER_T7_TODO.md) | Completed T7 board and below-gate evidence | T7 audit |
| [TEACHER_T8_PLAN.md](TEACHER_T8_PLAN.md) | Executed train-only single-action adaptive-routing successor | T8 audit |
| [TEACHER_T8_TODO.md](TEACHER_T8_TODO.md) | Completed T8 board and oracle-ceiling evidence | T8 audit |
| [TEACHER_T9_PLAN.md](TEACHER_T9_PLAN.md) | Executed train-only multi-action adaptive-routing successor | T9 audit |
| [TEACHER_T9_TODO.md](TEACHER_T9_TODO.md) | Completed T9 board and auxiliary-risk evidence | T9 audit |
| [TEACHER_T10_PLAN.md](TEACHER_T10_PLAN.md) | Executed conservative-risk margin-calibration successor | T10 audit |
| [TEACHER_T10_TODO.md](TEACHER_T10_TODO.md) | Completed T10 board and below-gate evidence | T10 audit |
| [TEACHER_T11_PLAN.md](TEACHER_T11_PLAN.md) | Active risk-penalized multi-action successor | all T11 work |
| [TEACHER_T11_TODO.md](TEACHER_T11_TODO.md) | Active T11 dependency board and exit gates | every T11 iteration |

Operational skill:
[`skills/manage-metricgan-research/SKILL.md`](skills/manage-metricgan-research/SKILL.md).

Current release state: **the corrected true-length S0 baseline v2 is promoted
and canonical. S0-WB selected epoch 34/stop 42 and S0-NB selected 41/49.
T1 failed calibration. T2 then restored exact official parity, but both
D2-OFFICIAL and D2-RANGE failed scalar/local-direction gates. T3 direct
perceptual direction passed locally, but every full E1/E2 proposal was harmful.
T4 found only `+0.002034`; T5 improved this to `+0.005075` but remained below
the gate and nearly exhausted SI-SDR margin. T6 selected exact T5. T7 gained
only `+0.004931`. T8 learned a safe `+0.009197` calibration router, but its
single-action oracle ceiling was only `+0.014197`. T9 raised the oracle to
`+0.031116`, but its PESQ-only decisions over-consumed SI-SDR; threshold
`0.02` retained `+0.017368` PESQ and missed SI-SDR by `0.029120` dB. T0
remains selected. T10 passed auxiliary guards but gained only `+0.008015` on
`val_select`; T11 now penalizes aggressive actions on fresh support**.

Open work:

- implement and validate the predeclared T11 risk-penalized action utility;
- run fresh-calibration/rank/select in order without reading test;
- no cache, student or test evaluation may start before teacher promotion;
- the TTS metric-critic hypothesis remains a separate future campaign.
