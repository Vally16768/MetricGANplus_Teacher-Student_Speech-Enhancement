# Official-teacher two-stage pilot audit

Run: `20260727-official-two-stage-pilot-s0-a1`  
Status: `pilot-passed`, independently audited, verification-only.

## Snapshot and scope

- clean commit: `0756a68f89b73dd31c6ce6265d7c56173d81a47a`;
- dataset: frozen read-only VoiceBank+DEMAND pilot manifests;
- support: 256 `train_fit`, 32 `val_rank`, 64 `val_select`, 64 test pairs;
- seed: 0;
- cells and hashed model packages: 7/7;
- reported sample files: 84/84;
- independent audit issues: 0;
- valid for promotion: false.

All manifest hashes remained unchanged. Pair and clean-identity overlap across
the four splits remained zero.

## Official teacher baseline

T0 reconstructed the pinned SpeechBrain MetricGAN+ checkpoint exactly and
produced:

| Split | PESQ-WB | STOI | SI-SDR | delta-SNR | Pairs |
|---|---:|---:|---:|---:|---:|
| `val_rank` | 2.7815 | 0.8723 | 5.8919 | -0.5175 | 32 |
| `val_select` | 2.8238 | 0.8867 | 6.4476 | +0.4818 | 64 |
| test | 3.2626 | 0.9266 | 6.1349 | -2.2728 | 64 |

The test PESQ-WB is consistent with the expected above-three behavior and
confirms that the earlier low teacher result came from the simplified local
training/frontend, not from this checkpoint reconstruction.

## Proxy and T1 behavior

The WB proxy used 384 training and 96 validation candidates. Held-out
calibration was strong on the fixed candidate distribution: MAE 0.3196,
Pearson 0.9539 and Spearman 0.9325.

That calibration did not make the frozen proxy safe as an unconstrained
generator objective. Both T1 branches moved away from the official checkpoint
and lost true PESQ-WB on `val_select`:

| Candidate | Epoch 0 | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|
| T1 control | 2.8238 | 2.3918 | 2.3541 |
| T1 frozen-proxy | 2.8238 | 2.4457 | 2.4285 |

The metric branch increased its predicted training PESQ while true validation
PESQ fell. This is direct proxy-exploitation evidence. Its SI-SDR increased,
showing that a multi-objective trade-off, not a failed optimizer, drove the
teacher away from the PESQ optimum. Best-checkpoint selection restored the
epoch-0 official weights for both branches.

The promotion gate failed its required PESQ gain. STOI and SI-SDR guardrails
passed after checkpoint restoration. Because this was a verification pilot,
the graph continued with an explicit override, but `T0-WB-OFFICIAL` remained
the downstream teacher. A full run must stop at the same gate.

## Student behavior

The fresh causal-max students learned within three pilot epochs, but this
support is not sufficient for an academic quality claim:

| Cell | Band / PESQ mode | Test PESQ | STOI | SI-SDR | delta-SNR |
|---|---|---:|---:|---:|---:|
| S0-WB | WB / WB | 2.8911 | 0.8804 | 4.0016 | -3.2172 |
| S0-NB | NB / NB | 3.3085 | 0.8529 | 4.5429 | -2.6757 |
| S1-WB | WB / WB | 2.8911 | 0.8804 | 4.0070 | -3.2131 |
| S1-NB | NB / NB | 3.3077 | 0.8527 | 4.5317 | -2.6826 |

Since T1 was rejected, S0 and S1 consumed the same teacher targets. Their
paired PESQ differences were -0.00007 WB and -0.00085 NB, consistent with
fresh-training variation rather than a teacher effect. The low STOI/SI-SDR
values remain explicit student guardrail warnings.

## Cache behavior

One content-addressed local cache served both stages because the accepted
teacher checkpoint did not change. It occupies 54 MiB for 256 training pairs,
stores FP16 teacher outputs, contains no noisy/clean dataset copies and records
both requested stage labels. It is below the ignored Desktop-local runtime
root, not on Kingston and not in Git.

## Conclusion and next gate

The seven-cell graph is operational and independently valid as an engineering
pilot. It is not promotable and does not justify full training.

Before another pilot, replace the unbounded frozen-proxy generator objective
with an official MetricGAN-style bounded target-score objective and add a
trust-region/refresh mechanism so newly generated examples cannot escape the
proxy calibration distribution. Validate the change with unit tests, a clean
GPU smoke and true `val_select` PESQ. Full training remains blocked until T1
gains at least 0.01 PESQ-WB without violating STOI or SI-SDR guardrails.
