# T16 fine-action quadratic router plan

Status: **predeclared**
Date: **2026-07-28**

## Rationale

T14 remains the best safe result at `+0.009365` PESQ on `val_select`. T15
showed that train-only OOF shrinkage does not repair the train-to-validation
shift (`+0.009070`). T14's selected policy frequently chose the coarse
`-0.4` and `-0.6` actions while operating close to the SI-SDR constraint.
T16 tests whether intermediate strengths can retain PESQ while spending the
auxiliary-metric budget more efficiently.

## Frozen protocol

- VoiceBank+DEMAND only; WB at 16 kHz; dataset read-only.
- Start from the official MetricGAN+ checkpoint and the existing clean-free
  feature/router implementation.
- Fit support is the exact 584 train-only pairs used by T13--T15.
- Action logit deltas are exactly
  `[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]`.
- Use the T14 152-feature quadratic transform and ridge grid
  `[0.1, 1, 10, 100, 1000]`; do not use T15 OOF shrinkage.
- Search the same 336 multi-objective policies on `val_rank` once:
  STOI weights `[0,1,2,4]`, SI-SDR weights `[0,.01,.02,.04]`,
  strength penalties `[0,.01,.02]`, thresholds
  `[0,.0025,.005,.0075,.01,.0125,.015]`.
- Require the strict rank pre-gate, freeze one checkpoint, then evaluate it
  once on `val_select`. Test remains unread.

## Gate

- PESQ-WB gain at least `+0.01`;
- STOI loss at most `0.002`;
- SI-SDR loss at most `0.25` dB;
- checkpoint round-trip and provenance contracts pass.

A point-estimate pass is only a candidate. Promotion and shutdown require
independent recomputation, paired uncertainty analysis, and the confirmation
protocol in the project skill.

## Terminal campaign rule

T16 is the final search experiment authorized in this campaign. If it fails
the unchanged gate, record and audit the negative result, then stop searching;
do not create T17 or another successor. If it passes, perform only the
predeclared independent confirmation needed for promotion.
