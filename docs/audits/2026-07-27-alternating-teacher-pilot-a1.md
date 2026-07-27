# Alternating-teacher pilot audit

Run: `20260727-alternating-teacher-pilot-s0-a1`  
Status: execution passed, independently audited, teacher gate failed,
verification-only and non-promotable.

## Integrity

- clean commit: `9ad2b85`;
- frozen VoiceBank+DEMAND support: 256 train, 32 `val_rank`, 64
  `val_select`, 64 test pairs;
- shared project environment and one NVIDIA GPU;
- 7/7 cells and seven hashed model packages;
- 84/84 reported sample files;
- independent audit issues: 0;
- WB cells used 16 kHz WB references and PESQ-WB; NB cells used 8 kHz NB
  references and PESQ-NB.

The pinned official T0 checkpoint produced PESQ-WB 2.8238 on `val_select` and
3.2626 with STOI 0.9266 on the 64-pair test support. This is the credible
baseline; the earlier locally trained low-PESQ teacher is not used.

## Teacher result

The control branch degraded monotonically:

| Branch | Epoch 0 | Epoch 1 | Epoch 2 | Selected |
|---|---:|---:|---:|---:|
| control | 2.8238 | 2.8093 | 2.7964 | epoch 0 |
| alternating MetricGAN+ | 2.8238 | 2.8226 | 2.8261 | epoch 2 |

The alternating branch gained only +0.00221 PESQ-WB on `val_select`, below the
predeclared +0.01 promotion threshold. Its STOI and SI-SDR guardrails passed.
On test, reported only after selection, it changed PESQ-WB by -0.02029, STOI
by +0.00058 and SI-SDR by +0.31796. The teacher gate therefore failed and
`T0-WB-OFFICIAL` remained the downstream teacher. There is no teacher
improvement claim and no authorization for a full run.

## Discriminator and replay

The warm-start discriminator used 384 training and 96 held-out records. Its
held-out Pearson/Spearman correlations were 0.8585/0.8827, but its PESQ MAE was
0.5328 and its predicted range, 1.35–3.20, did not cover the clean target 4.5.

Before each generator epoch, D completed current, historical and current
updates with true normalized PESQ labels. Current-output calibration then
degraded:

| Epoch | Current rows | Historical rows | PESQ MAE | Pearson |
|---|---:|---:|---:|---:|
| 1 | 32 | 6 | 1.5002 | 0.8831 |
| 2 | 32 | 13 | 1.7555 | 0.8211 |

The generator-facing predicted PESQ exceeded the calibrated range
(4.96 then 5.10). The T0 trust anchor prevented collapse, but the discriminator
was not accurate enough around current generator outputs to justify a longer
or full teacher run.

Replay occupied 6.9 MiB under the ignored Desktop-local run directory. It
contained 64 generated FP16 outputs plus JSON labels/indexes. Clean and noisy
audio remained external read-only references; neither input class was copied
to the cache and Kingston was not modified.

## Students and fallback

Because the teacher failed its gate, S1 used the same content-addressed T0
cache as S0. Test PESQ reproduced within numerical variation:

| Profile | S0 | S1 | Delta |
|---|---:|---:|---:|
| WB/PESQ-WB | 2.89136 | 2.89135 | -0.00002 |
| NB/PESQ-NB | 3.30959 | 3.30848 | -0.00111 |

This verifies fallback/cache behavior; it is not a teacher-effect result. The
WB student remains inadequate and must not be promoted from this pilot.

## Decision

Do not run this configuration at full scale. The next teacher experiment must
be teacher-only until its true `val_select` gate passes. It must improve
current-output discriminator calibration, use at least the original recipe's
100 current examples per generator epoch, and stop before regenerating caches
or training S1 students when the teacher gate fails. Test remains excluded
from tuning.

