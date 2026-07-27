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
| C11 | Run unit/integration suite | 39/39 tests passed, including official-teacher, failed-gate fallback and cache identity/FP16 contracts | passed |
| C12 | Run GPU end-to-end smoke | stable post-cleanup `...-a5`: six cells, six models, 36/36 samples, zero audit issues | passed |
| C13 | Run canonical full/pilot experiments | pilot passed; first full `...-a1` stopped during inadequate legacy WB student and is invalid; replacement rerun pending | failed |
| C14 | Audit and promote only valid outputs | pilot reconciles 0 issues but is non-promotable; stopped full documented and excluded; no full result | blocked |
| C15 | Remove legacy public surface after verified archive | authorized removal applied; recovery anchors `5129bae` + local archive | passed |
| C16 | Final docs/privacy/scope audit | project guard passed with zero issues after authorized cleanup | passed |
| C17 | Recover stronger MP-SENet-campaign student architecture | exact causal-max architecture traced; external sources read-only; mixed-dataset weights explicitly excluded | passed |
| C18 | Implement causal-max aliases for WB/NB | new names preserve 96x1 checkpoint semantics; configs and research plan updated | passed |
| C19 | Validate causal-max architecture | 39/39 tests, direct WB/NB CUDA backward and seven-cell A3 smoke/audit passed | passed |
| C20 | Rerun pilot/full with causal-max students | only after clean committed two-stage smoke passes; compare S1 versus S0 within profile | blocked |
| C21 | Explain low teacher PESQ against the original implementation | simplified 10-epoch teacher/frontend/loss mismatch isolated; official primary sources and checkpoint inspected | passed |
| C22 | Import the official MetricGAN+ WB teacher reproducibly | pinned revision/SHA-256; exact 512/256/512 Hamming log-magnitude frontend; 21/21 tensors; offline checkpoint round-trip | passed |
| C23 | Implement the two-stage teacher-effect campaign | T0 official → C0 → S0 WB/NB → T1 control/metric gate → C1 → fresh S1 WB/NB; seven-cell report/audit | passed |
| C24 | Validate local persistent teacher caches | Desktop-local, content-addressed, FP16 teacher outputs, no noisy/clean duplication, resume and quantization tests | passed |
| C25 | Run two-stage GPU smoke and independent audit | A3: clean commit; 7 cells/models, 42/42 samples, cache dedup, zero audit issues; verification-only | passed |
| C26 | Run monitored two-stage pilot | clean commit `0756a68`; 7 cells/models, 84/84 samples, zero audit issues; T1 PESQ gate failed and T0 fallback passed | failed |
| C27 | Replace unsafe teacher proxy optimization | bounded target-score loss, T0 trust anchor and official log-spectral features implemented; 41/41 tests; clean CUDA smoke pending | in-progress |
| C28 | Rerun clean smoke and monitored pilot | true `val_select` PESQ gain >= 0.01 with STOI/SI-SDR guardrails | blocked |
| C29 | Run and promote full multi-seed campaign | only after C28 passes; independent audit and article-ready report | blocked |

## Execution rule

The next item may start only when its upstream gate passes. A failed experiment
is recorded with its cause and remains non-canonical. Commit and push are
authorized only after tests and the project guard pass.
