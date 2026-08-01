# IEEE Review Response Evidence Pack

This repository-root package contains the audited material needed to prepare
the point-by-point IEEE reviewer response and to revise the manuscript
manually. It is not an edited manuscript and does not replace the official
reviewer text supplied by the journal.

## Primary file

- `MetricGAN_IEEE_Review_Response_Evidence_Pack.docx` — the standalone Word
  working document with proposed reviewer responses, manuscript-ready wording,
  audited tables, paired uncertainty, figures, references, provenance, and
  claim limitations.

Supporting project records remain in:

- `docs/ADDRESSED_REVIEW.md` — canonical point-by-point evidence ledger.
- `.agents/REVIEW_REVISION_TODO.md` — completed review campaign board.
- `.agents/VALIDATION.md` — validation and audit record.

## Evidence status

- All approved new training is complete; no additional training is active.
- The review matrix contains ten systems evaluated on the same 824 test
  utterances per matching bandwidth protocol.
- The uncertainty appendix contains 48 deterministic paired-bootstrap rows,
  using 10,000 paired utterance draws with seed `20260730`.
- WB results use 16-kHz clean references and PESQ-WB.
- NB results use 8-kHz clean references and PESQ-NB.
- PESQ-WB and PESQ-NB are never pooled or ranked as one scale.
- The validated algorithmic future-signal dependency is 10 ms for both student
  profiles; this is not a measured end-to-end latency result.
- The final repository suite passed 114/114 tests, campaign and research-plan
  validation, `git diff --check`, and the project guard with zero issues.

## How to use the Word pack

1. Copy the official reviewer wording from the journal above the corresponding
   proposed response when verbatim reviewer quotation is required.
2. Adapt the proposed response text to the journal response template.
3. Apply the recommended manuscript wording manually to the article.
4. Insert or redraw the supplied tables and figures without changing the
   audited values or mixing WB and NB protocols.
5. Recheck the abstract, methods, results, limitations, conclusion, tables,
   captions, and supplementary text for consistency.
6. Preserve the limitations below in both the response and the revised paper.

## Limitations that must remain explicit

- All student comparisons use one training seed. Paired utterance bootstrap
  intervals do not measure training-seed variability.
- No formal listening test or artifact-specific subjective evaluation was
  performed.
- No audited stateful chunk-by-chunk streaming implementation exists.
- No target-device peak memory, worst-case runtime, end-to-end latency, power,
  or energy measurement was performed.
- INT8 storage is a theoretical weight-only lower bound, not a validated
  quantized deployment.
- The matched-input teacher is a reference, not a separately trained NB
  teacher.
- Historical and current SI-SDR values must not be pooled when their protocols
  differ.

## Integrity

Word-pack SHA-256:
`16a7e5bf722291757ea545406590776489056921dc2d275a1a6df310d0276169`

The 15-page Word document was rendered through Microsoft Word and visually
inspected page by page. The publication-facing package contains no personal
paths, usernames, credentials, hostnames, IP addresses, mount names, or
server-specific orchestration details.
