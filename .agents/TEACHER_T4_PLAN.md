# T4 bounded true-PESQ trust-region plan

Status: **active — T4-A scalar calibration implementation**

## Cause

T3 established that PMSQE has the correct local direction around T0, including
after proposed updates, but one complete train epoch is too large. E1 and E2
both reduced true `val_rank` PESQ at learning rates `1e-6`, `5e-7` and
`2.5e-7`; all six proposals were rolled back and both selected checkpoints are
exactly T0. T4 therefore changes update scale, not the dataset or teacher
baseline.

## T4-A — exact uniform mask-logit calibration

The official generator already exposes a bounded mask-logit perturbation.
Adding one scalar `delta` to every output-bin bias is exactly equivalent and
can be stored in an ordinary checkpoint.

- freeze every parameter except the deterministic bias transformation;
- scan the predeclared deltas
  `[-0.10,-0.08,-0.06,-0.04,-0.02,-0.01,0,+0.01,+0.02,+0.04]`;
- evaluate every candidate on true WB metrics over `val_rank`;
- reject candidates crossing the STOI/SI-SDR guardrails;
- select maximum PESQ only on `val_rank`;
- evaluate only the selected candidate once on `val_select`;
- never read test and never train students.

The candidate advances only with at least `+0.01` PESQ-WB on `val_select`,
STOI loss at most `0.002` and SI-SDR loss at most `0.25 dB`.

## T4-B — conditional micro-step backtracking

Run only if T4-A is safe but below the teacher threshold.

1. Start each trajectory from exact T0.
2. Use E2-PMSQE as the primary direction with MR-STFT/SI-SDR as constraints,
   not as the dominant accumulated epoch loss.
3. Create checkpoints after `1, 4, 16, 64, 256` optimizer steps.
4. For every checkpoint, line-search interpolation coefficients
   `[1, 0.5, 0.25, 0.125, 0.0625]` between T0 and the proposal.
5. Rank only with true `val_rank` PESQ plus unchanged guardrails.
6. Stop the trajectory at the first unsafe horizon; never repeat a full
   harmful epoch.
7. Evaluate the single selected candidate once on `val_select`.

If T4-B also fails, diagnose gradient conflict explicitly and predeclare a
constrained-gradient T5; do not relax the +0.01 promotion gate.

## Exit and downstream rules

- A real improvement still requires the true `val_select` gate and independent
  artifact audit.
- Test remains reporting-only after promotion.
- C4/student training remains blocked until teacher promotion.
- Dataset and T0 cache stay read-only/local under the existing contracts.
