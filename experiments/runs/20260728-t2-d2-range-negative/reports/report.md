# D2-RANGE — negative result

Status: **failed final discriminator gate; T2 stopped**

The conditional train-only score-widening ablation used 7,000 candidates from
1,000 fixed train utterances: T0, noisy/T0/clean interpolations, local
five-percent directions and a bounded output mask. The fixed calibration and
200-record audit identities were unchanged.

Training selected epoch 6 and early-stopped at epoch 11 after LR reductions
from `5e-4` to `2.5e-4` and `1.25e-4`. The untouched audit obtained normalized
MAE `0.328655`, Pearson `0.728317` and Spearman `0.759910`. The 395 eligible
local comparisons obtained sign agreement `0.326582` and delta Spearman
`-0.622144`.

The selected checkpoint SHA-256 is
`e6726d500c168eac4880a2d3e0364e8b3283dc06597adf86f451720602e57239`.
It is evidence only and must not be used to guide a generator.

Per the predeclared stop rule, no T2 teacher update, C2 cache or fresh S2
student training was executed. The package excludes VoiceBank+DEMAND audio,
local FP16 candidates, teacher cache and training state.
