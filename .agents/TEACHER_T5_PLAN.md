# T5 true-PESQ frequency-curve search plan

Status: **active — predeclared zeroth-order successor**

## Cause

T4-A proved that a uniform negative mask-logit bias is safe and gives a small
true PESQ-WB gain, but only `+0.002034` on `val_select`. T4-B proved that the
PMSQE parameter gradient does not generalize: increasing step horizons
reduced `val_rank` PESQ, the 256-step full proposal became unsafe, and the
selected one-step interpolation changed `val_select` PESQ by `-0.000001`.

T5 removes the failed surrogate-gradient link. It optimizes true PESQ directly
in a deliberately low-dimensional, smooth and offline-loadable modification
of the official teacher.

## T5-A — frozen support

- source: only T3 `direction train` identities from VoiceBank+DEMAND;
- first 96 identities: coordinate-search fit;
- next 96 identities: untouched internal calibration;
- `val_rank`: selection among completed sweep checkpoints only;
- `val_select`: one selected candidate, one time;
- test: unread;
- dataset: external read-only; generated manifests stay local/ignored.

Fit, calibration, `val_rank` and `val_select` identities remain disjoint.

## T5-B — smooth frequency parameterization

Add a smooth 257-bin vector to the official
`mask_generator.linear2.bias`. The vector is the linear interpolation of eight
frequency knots over 0–8 kHz and is folded into an ordinary checkpoint.

- initialization: eight coefficients at `-0.10`, the safe T4-A solution;
- coefficient bounds: `[-0.20, +0.05]`;
- deterministic low-to-high coordinate order;
- true-PESQ coordinate steps: `0.08`, then `0.04`, then `0.02`;
- for each coordinate and step, evaluate the bounded `+step` and `-step`
  candidates on fit support;
- accept only the candidate with the highest fit PESQ if it also passes
  support-relative STOI/SI-SDR guardrails;
- evaluate calibration only after each complete eight-coordinate sweep.

This is zeroth-order optimization: no PESQ/PMSQE discriminator gradient and no
neural-network optimizer step.

## T5-C — selection and gate

Candidate set for `val_rank`:

1. exact T0;
2. uniform `-0.10`;
3. completed sweep at step `0.08`;
4. completed sweep at step `0.04`;
5. completed sweep at step `0.02`.

Reject candidates that lose more than `0.002` STOI or `0.25` dB SI-SDR on the
corresponding support. Select maximum true PESQ only on `val_rank`, then
evaluate that one checkpoint on `val_select`.

The teacher gate is unchanged:

- PESQ-WB gain at least `+0.01`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25` dB;
- production support only;
- test unread.

If the single-seed gate passes, independently re-evaluate the checkpoint,
confirm across three declared support seeds and require a paired PESQ
bootstrap confidence interval excluding zero before promotion or shutdown.

## Safety and interpretation

True-PESQ search can exploit the metric. STOI/SI-SDR guardrails are therefore
hard constraints, not secondary reporting. No student/cache/test work starts
from a single search result. If T5 fails, preserve the negative curve evidence
and predeclare a model-capacity successor; do not widen bounds or reuse
`val_select` for tuning.
