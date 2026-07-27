# Alternating-teacher smoke audit

Run: `20260727-alternating-teacher-smoke-s0-a2`  
Status: `smoke-passed`, independently audited, verification-only.

## Integrity and execution

- clean commit: `f5003ef`;
- frozen VoiceBank+DEMAND smoke support: 8 train, 4 `val_rank`, 4
  `val_select`, 4 test pairs;
- shared environment and one NVIDIA GPU;
- 7/7 cells, seven hashed generator/student model packages and 42/42 reported
  sample files;
- independent audit issues: 0;
- unchanged manifest hashes and zero split identity overlap.

The official teacher again produced test PESQ-WB 3.3407 and STOI 0.9334.

## Alternating discriminator

`T1-WB-METRIC` used the SpeechBrain four-convolution
spectral-normalized discriminator. Before the generator epoch it completed:

1. current clean/enhanced/noisy updates;
2. historical enhanced replay;
3. the same current clean/enhanced/noisy updates.

Clean targets were exactly 1. Enhanced/noisy targets were true normalized
PESQ. D was frozen during G. The D optimizer, checkpoint and refresh record
were saved in the resumable cell state.

The local replay contained two generated FP16 teacher outputs (344 KiB total),
one historical replay sample, label/index JSON and a reused noisy-score JSON.
It contained no noisy or clean dataset copy and produced no dataset write.

Two current examples and one D epoch are only a structural smoke. Accordingly,
current-output calibration was poor (MAE 0.784 PESQ, Pearson -0.817) and cannot
support a quality claim. The larger pilot must determine whether D calibrates
on meaningful support.

## Teacher gate

T0 `val_select` PESQ-WB was 3.0460. Both T1 branches restored their epoch-0
official checkpoint; the alternating branch's post-update value was 3.0456.
The required +0.01 PESQ gate therefore failed, while STOI/SI-SDR guardrails
passed. T0 was correctly reused downstream and the run remained
non-promotable.

The smoke validates execution, isolation, fallback and packaging. It does not
validate teacher improvement or authorize full training.
