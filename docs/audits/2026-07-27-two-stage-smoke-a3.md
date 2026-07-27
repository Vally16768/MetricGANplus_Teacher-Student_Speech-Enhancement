# Official-teacher two-stage smoke audit

Run: `20260727-official-two-stage-smoke-s0-a3`  
Status: `smoke-passed`, independently audited, verification-only.

## Snapshot and scope

- clean commit: `8d36d6223aa8377387dbf7f7c65ead425807a37d`;
- dataset: frozen read-only VoiceBank+DEMAND smoke manifests;
- seed: 0;
- cells: 7/7;
- hashed model packages: 7/7;
- reported sample files: 42/42;
- independent audit issues: 0;
- valid for promotion: false.

All four manifest hashes remained unchanged. Pair and clean-identity overlap
across `train_fit`, `val_rank`, `val_select` and `test` remained zero.

## Teacher behavior

The official T0 checkpoint produced PESQ-WB 3.0460 on four `val_select` pairs
and 3.3407 on four test pairs. Both one-epoch T1 candidates were worse on
`val_select` before best-checkpoint restoration:

| Candidate | Epoch-0 PESQ-WB | Epoch-1 PESQ-WB |
|---|---:|---:|
| T1 control | 3.0460 | 3.0374 |
| T1 PESQ proxy | 3.0460 | 3.0386 |

The teacher promotion gate therefore failed only its required PESQ gain. The
STOI and SI-SDR guardrails passed. Because this is a verification run, the
graph continued with an explicit override, but the downstream teacher remained
`T0-WB-OFFICIAL`. A full run would stop at this gate.

The smoke PESQ proxy used only 12 training and 6 validation candidates. Its
held-out Pearson correlation was negative and its prediction range collapsed;
this is expected to be inadequate evidence and cannot justify the T1 metric
branch. The pilot/full proxy calibration must be evaluated before interpreting
that branch.

## Cache behavior

The teacher cache identity includes checkpoint, frozen training manifest and
cache-contract hashes. It stores FP16 teacher waveforms/masks, has empty
noisy/clean cache fields and occupies 2.9 MiB for the smoke support.

Because T1 was rejected, S0 and S1 resolved to the same WB/NB cache manifests.
One physical cache recorded both requested stage labels. The resulting student
test PESQ differences were below 0.00004, consistent with a wiring comparison
in which the teacher did not change.

Earlier A1/A2 smoke directories remain preserved. A1 is superseded because a
failed gate used an ambiguous T1 label; A2 corrected the fallback label but
still duplicated identical caches by stage label. A3 validates both fixes.

## Conclusion

The complete seven-cell graph, bandwidth-matched evaluation, failed-gate
fallback, local cache reuse, report generation and package auditor work
end-to-end. This smoke is not statistical or publication evidence. The next
gate is a clean-snapshot pilot with larger proxy calibration and validation
support.
