# Academic reporting contract

## Evidence language

- `Observed`: present directly in an artifact.
- `Reproduced`: rerun from a verified snapshot under the same protocol.
- `Corrected`: rerun after a documented protocol/code correction.
- `Inferred`: indirect evidence; state the inference.
- `Unresolved`: exact provenance or validity is unavailable.

Never present an observed historical result as reproduced.

## Claim-to-evidence chain

```text
article claim
  -> canonical table/figure
  -> aggregation script + raw metrics
  -> predictions/support
  -> checkpoint + resolved config
  -> code commit + command + environment
  -> frozen manifest hashes
```

Each article table cell must be traceable through this chain.

## Article-ready package

A promoted study must provide:

- research question and hypothesis;
- architecture block and ablation variable;
- dataset/split protocol and limitations;
- declared seeds and model-selection rule;
- raw and aggregate metrics with uncertainty;
- per-domain results and support;
- compute and deployment measurements;
- failure/limitation analysis;
- reproducibility command and provenance;
- figure sources and generation scripts.

Maintain one canonical results table. Historical, invalidated and corrected
values may coexist only when clearly labeled and mapped.

The current canonical table and claim-to-artifact map are in
`docs/FINAL_RESULTS.md`.
