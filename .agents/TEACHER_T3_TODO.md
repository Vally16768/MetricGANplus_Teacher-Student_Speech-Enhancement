# T3 teacher-improvement TODO

Status: **active execution — clean full E1/E2 pilot pending**
Last update: **2026-07-28**  
Current phase: **T3.3 matched E1/E2 pilot**
Next action: **commit the audited trainer, then run the clean full pilot**

This board executes `TEACHER_T3_PLAN.md`. T1 and T2 remain immutable negative
evidence.

Allowed states: `pending`, `in-progress`, `blocked`, `passed`, `failed`,
`not-applicable`.

## T3.0 — Plan and boundaries

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.0.1 | Reconcile T2 terminal evidence | both D2 branches failed; zero G updates | passed |
| T3.0.2 | Freeze dataset/runtime/profile boundaries | VoiceBank read-only; WB teacher; CUDA/shared venv | passed |
| T3.0.3 | Predeclare E0/E1/E2/D3/E3 matrix | one control, direct loss, conditional rank critic | passed |
| T3.0.4 | Index plan and activate this board | index/skill/TODO synchronized; skill validation/guard pass | passed |

## T3.1 — Loss provenance and implementation

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.1.1 | Pin PMSQE source/revision/license | torch-pesq 0.1.2; core hashes match `3aac3c8`; MIT; CPU/CUDA finite gradients | passed |
| T3.1.2 | Implement MR-STFT/SI-SDR/anchor/PMSQE losses | isolated E1/E2 module; explicit WB/16 kHz and true-length contracts | passed |
| T3.1.3 | Add numerical/gradient/invariance tests | 8 focused + 77 full tests; real VoiceBank CUDA waveform/21-tensor gradients finite | passed |
| T3.1.4 | Calibrate train-only gradient weights | 16 train rows; anchor `4.30122085`; PMSQE `0.00186623`; final weights hash `41a47ff7...`; 21/21 parameter gradients | passed |
| T3.1.5 | Run full tests, config validation and guard | 78/78; plan/campaign valid; project guard zero issues | passed |

## T3.2 — Fixed local-direction support

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.2.1 | Freeze 1000/200/200 T3 identities | hash `04022b77...`; pair/clean disjoint from both T2 supports; audit zero issues | passed |
| T3.2.2 | Generate teacher-manifold candidates | 5,600 bounded mask-logit variants; zero-delta/cache MAE max `1.024e-05` | passed |
| T3.2.3 | Label candidates with true PESQ-WB | 5,600 FP16 local outputs with finite true PESQ-WB/PMSQE labels | passed |
| T3.2.4 | Audit PMSQE local direction | 771 eligible; sign `0.9222`; rho `0.8982`; min SNR quartile `0.8454`; gradients 21/21 | passed |
| T3.2.5 | Record E2 eligibility | every predeclared direct-loss gate passed | passed |

## T3.3 — Matched E1/E2 pilot

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.3.1 | Test rollback and exact resume | model/optimizer/scheduler/RNG equivalence; focused tests pass | passed |
| T3.3.2 | Run E1/E2 CUDA smoke | `20260728-t3-e1-e2-smoke-a1`; both branches accepted one update; verification-only | passed |
| T3.3.3 | Run monitored E1/E2 pilot | matched init/seed/order; immutable histories | in-progress |
| T3.3.4 | Apply `val_select` teacher gate | PESQ/STOI/SI-SDR and E2−E1 | blocked |
| T3.3.5 | Record direct-loss decision | pass, inconclusive or failed | blocked |

## T3.4 — Conditional D3/E3 rank critic

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.4.1 | Confirm D3 eligibility | E2 gate outcome matches plan condition | blocked |
| T3.4.2 | Implement pairwise D3 fitting | Huber + rank loss; resumable | blocked |
| T3.4.3 | Pass fixed local/finite-difference gate | sign/rank/SNR/current-gradient evidence | blocked |
| T3.4.4 | Run E3 smoke and pilot | frozen accepted D3; rollback active | blocked |
| T3.4.5 | Apply E3 teacher gate | true metrics vs E0 and E1 | blocked |

## T3.5 — Confirmation and students

| ID | Item | Evidence | Status |
|---|---|---|---|
| T3.5.1 | Confirm selected teacher over three seeds | paired mean/CI and guard metrics | blocked |
| T3.5.2 | Evaluate accepted T3 on test once | reporting-only record | blocked |
| T3.5.3 | Build content-addressed C3 | accepted hash; local FP16; no inputs | blocked |
| T3.5.4 | Train fresh S3-WB/S3-NB | bandwidth-correct max-50 runs | blocked |
| T3.5.5 | Generate final report and independent audit | T3/S3 deltas, weights, plots, provenance | blocked |
| T3.5.6 | Commit/push authorized artifacts | clean tested snapshot; no cache/dataset | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Opened T3 after final T2 discriminator failure | D2-OFFICIAL and D2-RANGE both failed local direction; no G/C2/S2 work | pin PMSQE implementation |
| 2026-07-28 | Froze and indexed the T3 method | E0/E1/E2 and conditional D3/E3 matrix; skill valid; project guard and canonical plan validator pass | T3.1.1 source/license audit |
| 2026-07-28 | Pinned and audited the direct perceptual dependency | torch-pesq 0.1.2 wheel/core hashes; upstream `3aac3c8`; MIT; shared-venv CPU/CUDA gradient smoke | implement E1/E2 loss stack |
| 2026-07-28 | Implemented the isolated E1/E2 loss stack | MR-STFT, true-length SI-SDR, official-frontend T0 anchor, WB-only PMSQE, train-only gradient calibration | complete numerical/CUDA validation |
| 2026-07-28 | Passed direct-loss numerical and model compatibility tests | 77/77 full suite; zero-delta mask parity; real VoiceBank CUDA PMSQE and all 21 parameter gradients finite | freeze train-only weights |
| 2026-07-28 | Froze fresh T3 support identities | 1,000/200/200; hash `04022b77...`; pair/clean disjoint from both T2 supports; audit zero issues | calibrate weights |
| 2026-07-28 | Calibration preflight stopped safely on a device mismatch | internal torch-pesq module stayed on CPU while candidate was CUDA; zero weights/model updates; CUDA regression added | validate and retry from clean fix |
| 2026-07-28 | Froze E1/E2 gradient weights on train only | 16 T3-train directions; anchor `4.30122085`, PMSQE `0.00186623`; hash `e9edaae1...`; 0 val/test rows | generate mask candidates |
| 2026-07-28 | Completed all fixed mask candidates | 5,600 FP16 variants; zero-delta/cache MAE max `1.024e-05`; true PESQ-WB and PMSQE labels complete | bind parameter-gradient evidence |
| 2026-07-28 | Parameter-gradient preflight stopped safely | cuDNN BLSTM backward requires train mode; no dropout, weights or model updates; fix isolated before audit | validate and retry gradient evidence |
| 2026-07-28 | Passed the untouched PMSQE direction gate | audit 771 eligible; sign `0.9222`; rho `0.8982`; min quartile `0.8454`; 21/21 parameter gradients finite | implement matched E1/E2 resume/rollback |
| 2026-07-28 | Implemented exact post-evaluation state and rollback | bit-exact model/optimizer/scheduler/RNG resume test; deterministic matched crops/order | run CUDA smoke |
| 2026-07-28 | Passed matched E1/E2 CUDA smoke | both branches trained/evaluated/selected; E2 current-direction gate passed; two-file metrics explicitly non-promotable | commit clean trainer and start full pilot |
