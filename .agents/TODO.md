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
| C14 | Audit and promote only valid outputs | converged S0 closure is active in `.agents/EXECUTION_TODO.md`; promotion awaits merged audit and resume repair | in-progress |
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
| C27 | Replace unsafe teacher proxy optimization | bounded target-score loss, T0 trust anchor, official log-spectral features, 41/41 tests and clean 7-cell CUDA smoke | passed |
| C28 | Rerun clean smoke and monitored pilot | smoke/pilot executed and audited; bounded proxy stable but true PESQ gate failed | failed |
| C29 | Run and promote full multi-seed campaign | only after a teacher-only successor to C31 passes; independent audit and article-ready report | blocked |
| C30 | Implement alternating MetricGAN+ discriminator refresh | SpeechBrain 4-conv D, clean=1 and true normalized noisy/enhanced labels, current/history/current updates, local FP16 replay, resumable D state; 45/45 tests and corrected clean A2 smoke/audit pass | passed |
| C31 | Validate alternating T1 in clean smoke/pilot | clean pilot `...-a1` audited 7/7 cells/models and 84/84 samples, but T1 gained only +0.00221 PESQ-WB and D current-output MAE degraded 1.50→1.76; full blocked | failed |
| C32 | Isolate the next teacher-only fidelity trial | execute `.agents/TEACHER_IMPROVEMENT_PLAN.md` only after execution-board P3 promotion; require 100 current examples/refresh, calibration gate and stop before C1/S1 on failure | blocked |
| C33 | Define the ordered three-phase program | phase 1 official T0→C0→S0; phase 2 metric-aware WB teacher gate; phase 3 C1→fresh S1 only after gate | passed |
| C34 | Implement an official-baseline-only campaign | `smoke/pilot/run-baseline`; exactly T0, S0-WB, S0-NB; subset-aware report/audit | passed |
| C35 | Validate baseline-only implementation | 46/46 tests; plan/guard pass; clean protocol metadata; A1 CUDA smoke audited 3/3 cells/models, 18/18 samples, zero issues and zero cached inputs | passed |
| C36 | Run the full official baseline | `...-a1` audited: 3/3 cells/models, 54/54 samples, zero issues; WB best epoch 20 and NB best epoch 18/stop 20 establish the continuation need | passed |
| C37 | Improve the WB enhancement teacher with a metric discriminator | teacher-only trials; true PESQ-WB/STOI/SI-SDR gate; no S1 or full run before gate | blocked |
| C38 | Retrain students from an accepted T1 teacher | new content-addressed C1; fresh S1-WB/S1-NB with S0-matched architecture/seed/schedule | blocked |
| C39 | Evaluate the TTS metric-discriminator hypothesis separately | select TTS generator/data, recalibrate metric on synthesis outputs and keep claims/provenance outside enhancement campaign | blocked |
| C40 | Replace the 20-epoch student ceiling | commit `330e501`; max 50, plateau factor 0.5/patience 2/min LR 1e-6, early stop patience 8; 48/48 tests and clean three-cell CUDA smoke with zero audit issues | passed |
| C41 | Continue ceiling-limited official students | run `...cont50...-a1` audited with 0 issues; WB best 34/stop 42, NB best 41/stop 49, both early-stopped and not ceiling-limited | passed |
| C42 | Freeze the deferred teacher-improvement protocol | indexed canonical plan with E0/E1/E2 ablations, discriminator calibration gate, teacher promotion gate and separate TTS boundary | passed |
| C43 | Track the ordered post-NB program iteratively | `.agents/EXECUTION_TODO.md` owns P1–P6 dependencies, evidence, one active item and next action; project skill enforces updates | passed |
| C44 | Repair post-evaluation resume robustness | deterministic interrupt/resume equivalence for LR, patience, best state and checkpoint; unit/guard/CUDA smoke | in-progress |

## Execution rule

The next item may start only when its upstream gate passes. A failed experiment
is recorded with its cause and remains non-canonical. Commit and push are
authorized only after tests and the project guard pass.
