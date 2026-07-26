# Contracts

## Evidence precedence

1. immutable checkpoint/config/raw metrics/predictions/manifest hashes;
2. verified code commit;
3. exact launch command and environment;
4. generated aggregate tables;
5. reports and architecture documents;
6. article prose;
7. current code that postdates a historical run.

## Run identity

Use `<YYYYMMDD>-<purpose>-<model>-s<seed>-<shortid>`. Never reuse an existing
directory.

Minimum `provenance.json`:

```json
{
  "schema_version": 1,
  "run_id": "20260726-student-s-s0-a1b2c3",
  "status": "planned",
  "git_commit": "<40-hex>",
  "git_dirty": false,
  "config_sha256": "<64-hex>",
  "seed": 0,
  "command": "<exact command>",
  "manifest_sha256": {},
  "initialization": {},
  "created_utc": "<ISO-8601>"
}
```

Create the exact runtime record under ignored `local/runs/`. Promote a sanitized
record to `experiments/runs/` only after validation. Keep the private mapping
between logical dataset IDs and machine paths outside Git.

Canonical validation additionally requires:

- `provenance/config_resolved.yaml`;
- `provenance/environment.txt`;
- one complete log;
- `metrics/summary.json`;
- sample-level metrics/predictions when available;
- at least one selected model;
- plots and report;
- `status.json` with `status: valid`.

## Artifact retention

| Artifact | Canonical Git | External/ignored |
|---|---|---|
| source/config/docs | yes | no |
| resolved config/provenance | yes | no |
| raw/aggregate metrics | yes | no |
| selected model/checkpoint | yes | optional mirror |
| teacher cache/audio | no | yes |
| dataset | no | yes |
| debug/smoke bulk | no | yes |
| failed/superseded bulk | no | temporary external evidence |
| failure lesson/audit row | yes | no |

## Validity states

- `planned`
- `running`
- `failed`
- `invalid`
- `evaluated`
- `audited`
- `valid`
- `superseded`

Only `valid` may be promoted. `superseded` may remain in an article comparison
only if the comparison is intentional and its provenance remains complete.
