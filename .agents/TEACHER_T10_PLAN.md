# T10 conservative-risk teacher-routing plan

Status: **predeclared — implementation pending**

## Cause and hypothesis

T9 established a real multi-action calibration oracle (`+0.031116` PESQ-WB),
but its PESQ-only router selected strong suppression too often. At its most
conservative frozen threshold (`0.02`) it retained `+0.017368` PESQ and passed
the final STOI guard (`-0.001773`), while SI-SDR missed the unchanged final
limit by only `0.029120` dB (`-0.279120` versus `-0.25`).

T10 does not refit the T9 regressors and does not relax the final teacher gate.
It calibrates a more conservative activation margin on fresh train-only
support, allowing exact T0 whenever predicted PESQ benefit is not large enough
to justify the suppression risk.

## Frozen inputs and fresh calibration

- official T0 teacher and VoiceBank+DEMAND WB/16 kHz only;
- exact four T9 actions and four frozen T9 ridge regressors;
- no new fit and no generator update;
- fresh calibration: first 128 identities from the T3 `audit` partition,
  disjoint from T9 fit and calibration;
- fixed margins: `[0.020, 0.0225, 0.025, 0.0275, 0.030, 0.035, 0.040]`;
- no clean reference at inference;
- test, teacher cache and students remain blocked.

## Pre-validation and final gates

Select the margin with highest calibration PESQ among candidates satisfying:

- PESQ-WB gain at least `+0.01`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25` dB;
- at least one non-base action;
- portable checkpoint round-trip.

This pre-gate now equals the final scientific guardrails; T9's stricter
exploration-only `0.0015/0.15` filter is not reused. If no margin is eligible,
T10 stops before validation.

An eligible T10 reads `val_rank` once. It reads `val_select` once only if rank
PESQ improves and both guardrails pass. The final promotion gate remains
`+0.01` PESQ-WB, STOI `>= -0.002`, SI-SDR `>= -0.25`, no test read. A single
passing result still requires independent recomputation, declared support
seeds, paired bootstrap confidence interval and package audit before promotion
or shutdown.
