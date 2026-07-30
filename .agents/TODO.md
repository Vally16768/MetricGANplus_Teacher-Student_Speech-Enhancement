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
| C14 | Audit and promote only valid outputs | canonical corrected v2 package audited and pushed at `65b9a9c`; exact v1 model hashes retained | passed |
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
| C32 | Isolate the next teacher-only fidelity trial | A3 executed two strict 100+100 refreshes and stopped before E1/E2/C1/S1 when calibration failed | failed |
| C33 | Define the ordered three-phase program | phase 1 official T0→C0→S0; phase 2 metric-aware WB teacher gate; phase 3 C1→fresh S1 only after gate | passed |
| C34 | Implement an official-baseline-only campaign | `smoke/pilot/run-baseline`; exactly T0, S0-WB, S0-NB; subset-aware report/audit | passed |
| C35 | Validate baseline-only implementation | 46/46 tests; plan/guard pass; clean protocol metadata; A1 CUDA smoke audited 3/3 cells/models, 18/18 samples, zero issues and zero cached inputs | passed |
| C36 | Run the full official baseline | `...-a1` audited: 3/3 cells/models, 54/54 samples, zero issues; WB best epoch 20 and NB best epoch 18/stop 20 establish the continuation need | passed |
| C37 | Improve the WB enhancement teacher with a metric discriminator | current-output calibration failed after the predeclared retry; zero G updates and no T1 checkpoint | failed |
| C38 | Retrain students from an accepted T1 teacher | new content-addressed C1; fresh S1-WB/S1-NB with S0-matched architecture/seed/schedule | blocked |
| C39 | Evaluate the TTS metric-discriminator hypothesis separately | select TTS generator/data, recalibrate metric on synthesis outputs and keep claims/provenance outside enhancement campaign | blocked |
| C40 | Replace the 20-epoch student ceiling | commit `330e501`; max 50, plateau factor 0.5/patience 2/min LR 1e-6, early stop patience 8; 48/48 tests and clean three-cell CUDA smoke with zero audit issues | passed |
| C41 | Continue ceiling-limited official students | run `...cont50...-a1` audited with 0 issues; WB best 34/stop 42, NB best 41/stop 49, both early-stopped and not ceiling-limited | passed |
| C42 | Freeze the deferred teacher-improvement protocol | indexed canonical plan with E0/E1/E2 ablations, discriminator calibration gate, teacher promotion gate and separate TTS boundary | passed |
| C43 | Track the ordered post-NB program iteratively | `.agents/EXECUTION_TODO.md` owns P1–P6 dependencies, evidence, one active item and next action; project skill enforces updates | passed |
| C44 | Repair post-evaluation resume robustness | commit `5c48415`; 52/52 tests; real CUDA fault-injection A4 reconciled LR, patience, best state, selected hash and history | passed |
| C45 | Make evaluation invariant to variable-length batch padding | per-utterance protocol regression-tested; corrected S0 v2 audited and pushed at `65b9a9c` | passed |
| C46 | Close the ordered P1–P6 program | final S0 table and negative T1 evidence audited and pushed at `a485306`; conditional P5 work marked not applicable | passed |
| C47 | Define a separate T2 teacher-successor protocol | indexed plan/TODO preserve T1, restore official batch-1 parity, fit D to convergence, add local directional gate and block students until teacher promotion | passed |
| C48 | Establish exact official discriminator parity | SpeechBrain v1.1.0 `36c180c`; exact frontend/model/labels/update order; 65/65 tests and CUDA gradient smoke passed | passed |
| C49 | Fit and audit a trustworthy D2 on current T0 outputs | D2-OFFICIAL and train-only D2-RANGE failed scalar/local gates; negative packages retained | failed |
| C50 | Run matched E1-control/E2-metric teacher pilots | blocked by final D2 failure; zero generator updates | blocked |
| C51 | Confirm and promote T2 across declared seeds | +0.01 PESQ-WB, STOI/SI-SDR guards, E2>E1, independent audit | blocked |
| C52 | Transfer an accepted T2 to fresh S2-WB/S2-NB | not run: no accepted T2 teacher or C2 cache exists | blocked |
| C53 | Define the T3 teacher successor | direct PMSQE branch plus conditional pairwise local critic; strict teacher/student gates | passed |
| C54 | Implement and audit T3 differentiable losses | pinned source, WB contracts, untouched local-direction gate and CUDA tests passed | passed |
| C55 | Run matched T3 teacher pilots | full E1/E2 produced six rollbacks and no accepted epoch; D3 ineligible after harmful E2 | failed |
| C56 | Confirm and transfer an accepted T3 | three seeds, C3, fresh S3-WB/S3-NB and final audit | blocked |
| C57 | Execute T4 bounded true-PESQ trust region | T4-A safe `+0.002034`; T4-B effectively T0; no promotion | failed |
| C58 | Execute T5 direct true-PESQ frequency-curve search | safe `+0.005075` PESQ but below gate; SI-SDR near limit | failed |
| C59 | Execute T6 affine-logit true-PESQ search | exact T5 selected; PESQ `+0.005075`, below gate | failed |
| C60 | Execute T7 confidence-conditioned true-PESQ search | safe `+0.004931` PESQ but below gate; SI-SDR near limit | failed |
| C61 | Execute T8 adaptive teacher routing | learned `+0.009197`; oracle `+0.014197` missed frozen `+0.015`; validation/test unread | failed |
| C62 | Execute T9 multi-action teacher routing | oracle `+0.031116`; PESQ-only decisions violated auxiliary guards; validation/test unread | failed |
| C63 | Execute T10 conservative-risk routing | select `+0.008015`; auxiliary guards passed; PESQ gate failed | failed |
| C64 | Execute T11 risk-penalized routing | rank `+0.010068`, select `+0.008349`; auxiliary guards pass but PESQ gate fails | failed |
| C65 | Execute T12 rank-selected risk policy | rank `+0.011176`, select `+0.008425`; auxiliary guards pass but PESQ gate fails | failed |
| C66 | Execute T13 multi-objective routing | rank `+0.010932`, select `+0.008806`; safe but below PESQ gate | failed |
| C67 | Execute T14 quadratic multi-objective routing | rank `+0.011275`, select `+0.009365`; safe but below PESQ gate | failed |
| C68 | Execute T15 cross-fitted quadratic calibration | rank `+0.011033`, select `+0.009070`; safe but below T14 and gate | failed |
| C69 | Execute T16 fine-action quadratic routing | rank `+0.010967`, select `+0.009619`; safest best but below gate | failed |
| C70 | Apply terminal T16 rule | campaign closed with no T17; test unread; no shutdown | passed |

## Execution rule

The next item may start only when its upstream gate passes. A failed experiment
is recorded with its cause and remains non-canonical. Commit and push are
authorized only after tests and the project guard pass.

## Active IEEE-review revision

The teacher-search line remains closed at T16. Reviewer-requested student
ablations, current-protocol baselines, paired utterance uncertainty, streaming
analysis and publication corrections are tracked exclusively in
`.agents/REVIEW_REVISION_TODO.md`. Additional complete-model seeds are outside
the active scope and remain a publication limitation.
