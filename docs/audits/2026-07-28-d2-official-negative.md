# D2-OFFICIAL negative audit

Status: **observed negative evidence; not accepted for generator guidance**  
Run: `20260728-t2-d2-official-s0-a1`

## Outcome

The exact-parity, batch-size-one `D2-OFFICIAL` discriminator completed its
predeclared stopping rule and failed both mandatory gates. No teacher
generator update was authorized.

The selected checkpoint came from epoch 1 and has SHA-256
`34112ac7c200245588f0b8832565b883edc8040ae277d032c954690b525b7a66`.
Training stopped after epoch 6 with early-stopping patience 5.

## Untouched 200-record audit

| Metric | Required | Observed | Result |
|---|---:|---:|---|
| normalized PESQ MAE | ≤ 0.06 | 0.289481 | fail |
| raw PESQ MAE | ≤ 0.30 | 1.447405 | fail |
| Pearson | ≥ 0.80 | 0.762561 | fail |
| Spearman | ≥ 0.80 | 0.776770 | fail |
| predicted range | target ±0.30 | 0.4864–4.3282 vs 1.5616–4.3050 | fail |

The independent local audit formed 400 controlled pairs around T0, of which
395 exceeded the true-delta floor. Sign agreement was `0.529114` against a
minimum of `0.70`; delta Spearman was `-0.492946` against a minimum of `0.60`.

## BatchNorm diagnostic

The same frozen checkpoint was evaluated once with official-style current
batch statistics. Normalized MAE improved from `0.289481` to `0.110310`, but
Pearson fell to `0.578196` and Spearman to `0.585754`. This cannot satisfy the
gate and does not repair the opposite local ranking. It is therefore not used
to relabel or promote the failed run.

## Decision

The failure is eligible for the single predeclared `D2-RANGE` ablation because
exact official parity remains validated while score coverage and local
ranking fail. D2-RANGE may use only train-derived interpolation and bounded
output variants. The original calibration/audit identities and thresholds
remain fixed. If D2-RANGE fails, teacher and student work stop and this
negative result is retained for the article.
