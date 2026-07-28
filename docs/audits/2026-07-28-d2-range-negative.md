# D2-RANGE final negative audit

Status: **observed final T2 discriminator failure**

## Support integrity

`20260728-t2-d2-range-support-s0-a1` was generated from the fixed D2 train
partition only. Its support SHA-256 is
`3d5632169c947c25dc234b17b741f4e5fd4fe9ea890cf59ecf9e495a6df16e05`.
The independent audit reconciled:

- 7,000 candidates across seven declared candidate types;
- 1,000 unique train parents;
- the same 200 untouched audit identities;
- zero missing candidates, wrong dtypes or non-finite labels;
- no copied noisy/clean source directories.

The observed train-only candidate PESQ-WB range was `1.0363–4.5366`, with
mean `2.8973` and standard deviation `0.7647`.

## Training and gates

D2-RANGE used the same architecture, optimizer, maximum 20 epochs, scheduler,
early-stopping rule and audit thresholds as D2-OFFICIAL. It selected epoch 6,
stopped at epoch 11 and produced checkpoint
`e6726d500c168eac4880a2d3e0364e8b3283dc06597adf86f451720602e57239`.

| Metric | Required | D2-OFFICIAL | D2-RANGE | Result |
|---|---:|---:|---:|---|
| normalized MAE | ≤ 0.06 | 0.289481 | 0.328655 | fail |
| Pearson | ≥ 0.80 | 0.762561 | 0.728317 | fail |
| Spearman | ≥ 0.80 | 0.776770 | 0.759910 | fail |
| local sign agreement | ≥ 0.70 | 0.529114 | 0.326582 | fail |
| local delta Spearman | ≥ 0.60 | -0.492946 | -0.622144 | fail |

D2-RANGE did not improve the accepted target; it degraded all five primary
fidelity/local-guidance measures relative to D2-OFFICIAL.

## Stop decision

The single allowed score-widening ablation is exhausted. The critic is unsafe
for generator guidance, so E1/E2 teacher pilots, three-seed teacher
confirmation, C2 generation and S2-WB/S2-NB training are not applicable.
There is no T2 teacher or student improvement claim.

Both failed checkpoint packages remain in `experiments/runs/` as reproducible
negative evidence for the article. Dataset audio and local regenerable caches
remain outside Git.
