# Official T0 student baseline full-run audit

## Scope and provenance

Run `20260727-official-baseline-full-s0-a1` executed the predeclared
three-cell baseline from clean commit `357c1df`: pinned official
`T0-WB-OFFICIAL`, `S0-WB`, and `S0-NB`. The students used the same causal-max
architecture and the same teacher ancestry, with bandwidth-specific outputs,
references, and PESQ protocols.

The frozen VoiceBank+DEMAND manifests contained 9,754 `train_fit`, 128
`val_rank`, 1,690 `val_select`, and 824 test pairs. The split audit found zero
pair or clean-reference overlap. The content-addressed Desktop-local teacher
cache contained 39,016 FP16 payloads (approximately 1.8 GiB) and no copies of
noisy or clean dataset inputs.

## Independently reconciled results

The package audit reconciled three cells, three hashed selected models, and
54 reported sample files with zero issues. Selection used `val_select`; test
was evaluated only after checkpoint selection.

| Cell | Protocol | Best / stop epoch | `val_select` PESQ | STOI | SI-SDR | Test PESQ | STOI | SI-SDR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T0-WB-OFFICIAL | WB, 16 kHz, PESQ-WB | 0 / 0 | 2.698864 | 0.861746 | 5.403473 | 3.122664 | 0.931190 | 8.482982 |
| S0-WB | WB, 16 kHz, PESQ-WB | 20 / 20 | 2.596915 | 0.853251 | 5.366443 | 3.050537 | 0.928756 | 8.720695 |
| S0-NB | NB, 8 kHz, PESQ-NB | 18 / 20 | 3.192184 | 0.850458 | 5.389149 | 3.613544 | 0.929136 | 8.913116 |

PESQ-WB and PESQ-NB are different protocols and the two student PESQ values
must not be compared as one metric.

## Convergence verdict

Execution and artifact integrity passed, but the original student schedule
does not establish convergence:

- `S0-WB` selected epoch 20, exactly the former maximum, so it is
  ceiling-limited.
- `S0-NB` selected epoch 18 and stopped at epoch 20 only because the former
  maximum was reached; this is insufficient to satisfy the new eight-check
  early-stopping rule.

These selected models remain valid historical 20-epoch observations, but are
not final converged student baselines. Continue both in a new immutable
two-cell package from their complete epoch-20 training states. The continuation
must preserve optimizer, scheduler, AMP scaler, history, best score, source
model hash, and source state hash; use a 50-epoch maximum,
`ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)`, and early stopping
after eight validation checks without improvement.
