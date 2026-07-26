# Canonical campaign TODO

Status values: `pending`, `in-progress`, `blocked`, `passed`, `failed`.
Update this register whenever a gate changes state.

| ID | Work item | Gate / evidence | Status |
|---|---|---|---|
| C01 | Inventory tracked, untracked and historical material | cleanup audit with keep/archive/remove-later class | passed |
| C02 | Protect local historical imports from accidental Git inclusion | root `.gitignore`; history remains readable | passed |
| C03 | Remap stale dataset prefixes only into ignored local manifests | 12,396 bound rows; zero missing/duplicate/overlap; source hashes retained | passed |
| C04 | Define one VoiceBank-only campaign entry point | `campaign.py validate/smoke-all/run-all` | passed |
| C05 | Implement WB teacher baseline/metric pair | same anchor/seed/schedule; WB proxy; GPU smoke | passed |
| C06 | Implement one WB teacher cache with WB and NB targets | dual cache manifests consumed in GPU smoke | passed |
| C07 | Implement WB student baseline/metric pair | WB profile/proxy; GPU smoke | passed |
| C08 | Implement NB student baseline/metric pair | NB profile/proxy; GPU smoke | passed |
| C09 | Implement true-metric evaluation and aggregation | profile metadata/support in canonical CSV/JSON | passed |
| C10 | Implement plots and academic report | curves, calibration, deltas, hashes, report and independent auditor | passed |
| C11 | Run unit/integration suite | 30/30 tests passed, including WB/NB ERB extraction | passed |
| C12 | Run GPU end-to-end smoke | stable post-cleanup `...-a5`: six cells, six models, 36/36 samples, zero audit issues | passed |
| C13 | Run canonical full/pilot experiments | pilot mode/manifests defined; requires clean committed snapshot | in-progress |
| C14 | Audit and promote only valid outputs | smoke reconciles 0 issues but is explicitly non-promotable; no full result | blocked |
| C15 | Remove legacy public surface after verified archive | authorized removal applied; recovery anchors `5129bae` + local archive | passed |
| C16 | Final docs/privacy/scope audit | project guard passed with zero issues after authorized cleanup | passed |

## Execution rule

The next item may start only when its upstream gate passes. A failed experiment
is recorded with its cause and remains non-canonical. Commit and push are
authorized only after tests and the project guard pass.
