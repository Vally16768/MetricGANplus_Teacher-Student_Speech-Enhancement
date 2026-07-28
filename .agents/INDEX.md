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
| [TEACHER_T11_PLAN.md](TEACHER_T11_PLAN.md) | Executed risk-penalized multi-action successor | T11 audit |
| [TEACHER_T11_TODO.md](TEACHER_T11_TODO.md) | Completed T11 board and below-gate evidence | T11 audit |
| [TEACHER_T12_PLAN.md](TEACHER_T12_PLAN.md) | Executed rank-selected risk-policy successor | T12 audit |
| [TEACHER_T12_TODO.md](TEACHER_T12_TODO.md) | Completed T12 board and below-gate evidence | T12 audit |
| [TEACHER_T13_PLAN.md](TEACHER_T13_PLAN.md) | Executed train-only multi-objective router | T13 audit |
| [TEACHER_T13_TODO.md](TEACHER_T13_TODO.md) | Completed T13 board and below-gate evidence | T13 audit |
| [TEACHER_T14_PLAN.md](TEACHER_T14_PLAN.md) | Executed quadratic multi-objective successor | T14 audit |
| [TEACHER_T14_TODO.md](TEACHER_T14_TODO.md) | Completed T14 board and below-gate evidence | T14 audit |
| [TEACHER_T15_PLAN.md](TEACHER_T15_PLAN.md) | Executed cross-fitted quadratic calibration | T15 audit |
| [TEACHER_T15_TODO.md](TEACHER_T15_TODO.md) | Completed T15 board and below-gate evidence | T15 audit |
| [TEACHER_T16_PLAN.md](TEACHER_T16_PLAN.md) | Executed terminal fine-action quadratic search | T16 audit |
| [TEACHER_T16_TODO.md](TEACHER_T16_TODO.md) | Completed terminal T16 board | canonical closure |

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
remains selected. T10 gained `+0.008015`; T11 gained `+0.008349`; T12 used
`val_rank` correctly but reached only `+0.008425`; T13 improved this to
`+0.008806`; T14 improved this to `+0.009365` on `val_select`; T15 OOF
calibration regressed to `+0.009070`. Terminal T16 reached the best safe
result, `+0.009619`, but remained `0.000381` below the gate; search is closed
without T17**.

Open work:

- preserve T16 and earlier negative evidence for the article;
- do not continue teacher search without a new explicit research mandate;
- no cache, student or test evaluation may start before teacher promotion;
- the TTS metric-critic hypothesis remains a separate future campaign.
