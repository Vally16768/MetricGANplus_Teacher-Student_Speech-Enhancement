# T4 teacher-improvement TODO

Status: **complete negative outcome — no T4 teacher promoted**
Last update: **2026-07-28**
Current phase: **T4.3 closed; T5 true-PESQ curve search activated**
Next action: **preserve T4 evidence and execute `.agents/TEACHER_T5_TODO.md`**

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
| T4.1.4 | Create immutable run contract | commit `e58f8b1`; contract validation passed | passed |
| T4.1.5 | Scan fixed grid on true `val_rank` | `...t4-logit-bias...a1`; selected `-0.10`; all ten complete | passed |
| T4.1.6 | Evaluate one selected candidate on `val_select` | PESQ `+0.002034`; STOI `-0.000467`; SI-SDR `-0.161412`; no test | failed |
| T4.1.7 | Independently re-evaluate/audit a passed candidate | gate did not pass; no promotion audit applicable | not-applicable |

## T4.2 — Conditional micro-step backtracking

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.2.1 | Confirm T4-B eligibility | T4-A passed both guardrails but gained only `+0.002034` PESQ | passed |
| T4.2.2 | Implement exact micro-step/resume trajectory | horizons `1,4,16,64,256`; each incomplete horizon restarts exact T0 | passed |
| T4.2.3 | Implement checkpoint interpolation line search | alpha `1,.5,.25,.125,.0625`; true `val_rank` gate | passed |
| T4.2.4 | Run focused/full tests and CUDA smoke | 85/85; `...t4b...smoke...a1` complete; finite one-step/checkpoints/eval | passed |
| T4.2.5 | Run contracted T4-B pilot | all 5 horizons; 25 rank candidates; horizon 256 unsafe | passed |
| T4.2.6 | Apply one-shot `val_select` gate and audit | selected H1/a=.125; PESQ `-0.000001`; gate failed; no test | failed |

## T4.3 — Exit or successor

| ID | Item | Evidence | Status |
|---|---|---|---|
| T4.3.1 | Confirm teacher improvement across declared seeds | positive mean and paired PESQ CI excludes zero | blocked |
| T4.3.2 | Promote accepted teacher evidence | independent package/privacy/hash audit | blocked |
| T4.3.3 | Stop and shut down after genuine success | immediate power recheck; no active writes | blocked |
| T4.3.4 | Predeclare T5 if T4-B fails | true-PESQ low-dimensional curve search; unchanged gate | passed |

## Progress log

| Date | Change | Evidence | Next action |
|---|---|---|---|
| 2026-07-28 | Opened T4 after terminal T3 rollback result | T3 A2 selected exact T0 for E1/E2; no teacher improvement | implement T4-A |
| 2026-07-28 | Implemented bounded scalar checkpoint calibration | fixed ten-delta grid; true WB rank/select separation; focused equivalence/bound tests pass | full validation and clean commit |
| 2026-07-28 | Passed complete pre-run validation | 83/83 tests; research plan and real campaign validation pass; project guard zero issues | commit and create run contract |
| 2026-07-28 | Closed T4-A below the promotion threshold | selected `-0.10`; `val_select` PESQ `+0.002034`; both quality guardrails pass; test unread | activate T4-B |
| 2026-07-28 | Implemented deterministic T4-B micro-step/backtracking | exact T0 per horizon; atomic horizon resume; ordinary checkpoints; 85/85 tests and validators pass | clean commit and CUDA smoke |
| 2026-07-28 | Passed clean T4-B CUDA smoke | one T0 micro-step; alpha `1/.5`; two-file rank/select path; finite artifacts; explicitly verification-only | full contracted pilot |
| 2026-07-28 | Closed full T4-B as negative | selected H1/a=.125; `val_select` PESQ `-0.000001`; 256-step proposal unsafe; no test/cache/students | activate T5 true-PESQ curve search |
