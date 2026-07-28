# T4 teacher-improvement TODO

Status: **in progress — T4-A clean snapshot preparation**
Last update: **2026-07-28**
Current phase: **T4.1 bounded scalar calibration**
Next action: **commit the tested snapshot and create the immutable T4-A run contract**

This board executes `TEACHER_T4_PLAN.md`. T0 remains the selected teacher; T1,
T2 and T3 remain immutable negative evidence.

Allowed states: `pending`, `in-progress`, `blocked`, `passed`, `failed`,
`not-applicable`.

## T4.0 — Boundaries and cause

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.0.1 | Reconcile terminal T3 evidence | six harmful proposals; E1/E2 selected exact T0 | passed |
| T4.0.2 | Freeze dataset/runtime/profile boundaries | VoiceBank read-only; WB/16 kHz teacher; CUDA/shared venv | passed |
| T4.0.3 | Predeclare T4-A/T4-B and unchanged gate | bounded scalar scan then conditional micro-step backtracking | passed |
| T4.0.4 | Index this board as active source of truth | index, skill, TODO and documentation register synchronized | passed |

## T4.1 — Exact uniform mask-logit calibration

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.1.1 | Implement bounded bias folding | ordinary checkpoint; no runtime wrapper; `+/-0.10` bound | passed |
| T4.1.2 | Add focused equivalence and bound tests | folded bias equals official mask-logit variant | passed |
| T4.1.3 | Run full tests, validators and project guard | 83/83; plan/campaign valid; guard and diff check pass | passed |
| T4.1.4 | Create immutable run contract | exact commit/config/T0/T3 baseline ancestry | pending |
| T4.1.5 | Scan fixed grid on true `val_rank` | ten predeclared deltas; WB PESQ/STOI/SI-SDR | pending |
| T4.1.6 | Evaluate one selected candidate on `val_select` | no test read; one-shot gate | pending |
| T4.1.7 | Independently re-evaluate/audit a passed candidate | exact checkpoint/hash/support and metric reconciliation | blocked |

## T4.2 — Conditional micro-step backtracking

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.2.1 | Confirm T4-B eligibility | T4-A safe but gain below `+0.01` | pending |
| T4.2.2 | Implement exact micro-step/resume trajectory | horizons `1,4,16,64,256`; exact T0 restart | pending |
| T4.2.3 | Implement checkpoint interpolation line search | alpha `1,.5,.25,.125,.0625`; true `val_rank` gate | pending |
| T4.2.4 | Run focused/full tests and CUDA smoke | finite direction, rollback and reproducibility | pending |
| T4.2.5 | Run contracted T4-B pilot | stop at first unsafe horizon; no full harmful epoch | pending |
| T4.2.6 | Apply one-shot `val_select` gate and audit | gain `>=.01`; STOI/SI-SDR guards; no test | pending |

## T4.3 — Exit or successor

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.3.1 | Confirm teacher improvement across declared seeds | positive mean and paired PESQ CI excludes zero | blocked |
| T4.3.2 | Promote accepted teacher evidence | independent package/privacy/hash audit | blocked |
| T4.3.3 | Stop and shut down after genuine success | immediate power recheck; no active writes | blocked |
| T4.3.4 | Predeclare T5 if T4-B fails | diagnose gradient conflict; do not relax gate | blocked |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Opened T4 after terminal T3 rollback result | T3 A2 selected exact T0 for E1/E2; no teacher improvement | implement T4-A |
| 2026-07-28 | Implemented bounded scalar checkpoint calibration | fixed ten-delta grid; true WB rank/select separation; focused equivalence/bound tests pass | full validation and clean commit |
| 2026-07-28 | Passed complete pre-run validation | 83/83 tests; research plan and real campaign validation pass; project guard zero issues | commit and create run contract |
