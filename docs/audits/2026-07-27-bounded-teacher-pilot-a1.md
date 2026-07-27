# Bounded-teacher pilot audit

Run: `20260727-bounded-teacher-pilot-s0-a1`  
Status: `pilot-passed`, independently audited, teacher gate failed,
verification-only.

## Integrity

- clean commit: `33ef895`;
- frozen VoiceBank+DEMAND support: 256 train, 32 `val_rank`, 64
  `val_select`, 64 test pairs;
- 7/7 cells and hashed model packages;
- 84/84 reported sample files;
- unchanged manifests and zero cross-split identity overlap;
- independent audit issues: 0.

## Teacher result

T0 reproduced PESQ-WB 2.8238 on `val_select` and 3.2626 on test. The bounded
branches no longer collapsed, but neither improved true PESQ:

| Branch | Epoch 0 | Epoch 1 | Epoch 2 | Best accepted |
|---|---:|---:|---:|---:|
| control | 2.8238 | 2.8093 | 2.7964 | epoch 0 |
| bounded frozen proxy | 2.8238 | 2.8197 | 2.8131 | epoch 0 |

The proxy itself remained well calibrated on fixed held-out candidates:
384/96 records, MAE 0.3196, Pearson 0.9539 and Spearman 0.9325. The metric
branch moved less than the control, proving that the bounded objective and T0
anchor constrain optimization. They do not establish a positive teacher gain.

The promotion gate failed only its required PESQ gain. Both best checkpoints
were the original T0 weights, so the downstream teacher remained
`T0-WB-OFFICIAL`.

## Students and cache

S0 test PESQ was 2.8913 WB and 3.3085 NB. Since T1 was rejected, S1 used the
same content-addressed local T0 cache. S1-S0 PESQ deltas were +0.00006 WB and
-0.00013 NB, consistent with fresh-training/numerical variation rather than a
teacher effect.

No dataset audio was copied into the cache, no Kingston path was written and no
runtime artifact entered Git.

## Decision

The bounded frozen-proxy variant is stable but scientifically negative on this
pilot. Do not run it at full scale and do not tune against test results.

The next T1 implementation must follow the defining MetricGAN+ behavior that is
still absent here: alternate discriminator and generator updates, label current
enhanced/noisy examples with true normalized PESQ, and replay historical
enhanced examples. It requires its own tests and clean smoke/pilot gate before
full training.
