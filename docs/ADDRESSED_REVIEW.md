# Addressed review — IEEE draft v2

Status: **working ledger; baseline evidence validated, training matrix in progress**
Draft reviewed: `MetricGAN_Teacher_Student_IEEE_Draft_v2.pdf` (6 pages,
created 2026-07-30)

This document distinguishes completed evidence from planned work. A planned
experiment is not an addressed criticism.

| Reviewer concern | Current assessment | Action/evidence | Article change | Status |
|---|---|---|---|---|
| Distillation benefit not isolated | Valid major concern | Run the frozen WB loss-component matrix in `.agents/REVIEW_REVISION_TODO.md` | Add an ablation table and restrict causal attribution to measured contrasts | in progress |
| NB teacher has privileged >4-kHz information | Valid information mismatch | Matched-input teacher reference completed on 824 pairs; NB clean-only student still required | Rewrite target-generation text and limitations | in progress |
| Reported 16-ms lookahead | Incorrect mathematical dependency | Correct model contract to 10 ms and add numerical dependency/streaming validation; distinguish library buffering | Replace all 16-ms claims and Table V entries | pending |
| NB results lack baselines | Valid major concern | Current-protocol noisy NB and matched-input teacher reference are validated; clean-only NB remains | Expand current-results table within NB protocol | in progress |
| One seed | Valid major concern | Complete D1 seeds 0, 1001, 2002 for both bandwidths; sample-level baseline evidence now exists for paired bootstrap | Report mean, standard deviation and 95% CIs | in progress |
| Reproducibility identifiers absent | Evidence exists but is missing from draft | Publish sanitized checkpoint, code/config, manifest and speaker-split hashes | Add reproducibility subsection/table | pending |
| Foundational references absent | Valid presentation/scholarship concern | Add original MetricGAN, MetricGAN+, VoiceBank+DEMAND, PESQ, STOI, SI-SDR and KD references | Replace the artificial 2024--2026 restriction | pending |
| Same FFT sizes at 16/8 kHz | Valid design asymmetry | State exact time supports; add duration-matched-resolution ablation only if compute budget permits | Justify or label as limitation; do not claim temporal equivalence | pending |
| “Energy” in Eq. (2) | Terminology is incorrect for summed magnitude | Rename to ERB-band magnitude or square the quantity only in a separately trained method | Correct equation prose and symbols | pending |
| Complexity incomplete | Valid | Add teacher neural-core MAC/s and explicit recurrent/buffer/STFT memory accounting; retain non-benchmark caveat | Expand complexity table and limitations | pending |
| No subjective/artifact evaluation | Not currently measured | Provide sanitized audio/examples only if publication packaging permits; do not invent listening evidence | Keep as explicit limitation/future work | open limitation |
| Figure/visual density | Valid | Replace Fig. 1 with a real training/inference block diagram; add ablation/seed plots and spectrogram examples | Revise figures and shorten abstract | pending |
| Historical/refinement discussion | Too long and weakly evidenced in draft | Retain only concise, traceable negative-result statement or remove | Shorten Sections V-A/V-C | pending |
| Historical SI-SDR discrepancy | Protocols differ and need explicit causal explanation | Identify differences in teacher, architecture, segmentation and evaluation; never pool values | Add a short non-comparability note or remove historical table | pending |

## Corrections already established by audit

- The student recurrent graph is frame-causal.
- With a centered 20-ms nonzero Hamming window, mathematical future signal
  dependence is 10 ms, not `n_fft/2 = 16 ms`. Library padding/buffering and
  measured end-to-end delay are separate quantities.
- The NB cached target is derived from a WB-teacher output and therefore can
  contain decisions conditioned on frequencies unavailable to the NB student.
  Resampling aligns the target representation, not the teacher information.
- The promoted seed-0 S0 package is valid one-seed evidence; it does not
  establish training variability or statistical significance.

## Validated current-protocol references

The reporting-only evaluation
`local/runs/20260730-review-baselines-test-s0-a1` used the untouched 824-pair
test manifest after all model-selection decisions. The three sample-level CSV
hashes, row counts and aggregate PESQ, STOI, SI-SDR and delta-SNR means were
independently reconciled within `1e-6`.

| System | Protocol | Pairs | PESQ | STOI | SI-SDR (dB) |
|---|---|---:|---:|---:|---:|
| Noisy input | WB, 16 kHz, PESQ-WB | 824 | 1.9701 | 0.9210 | 8.4454 |
| Noisy input | NB, 8 kHz, PESQ-NB | 824 | 2.9481 | 0.9212 | 8.4318 |
| Fixed WB teacher with matched 8-kHz-bandlimited input | NB, 8 kHz, PESQ-NB | 824 | 3.6058 | 0.9283 | 9.2583 |

These rows must not be compared across PESQ modes. The matched-input teacher
removes access to frequencies above 4 kHz at its input, but it is not a
dedicated narrowband model and does not replace the pending clean-only NB
student ablation.

## Evidence locations

- Canonical seed-0 package:
  `experiments/runs/20260727-converged-s0-baseline-v2/`
- Full current-protocol reference evidence:
  `local/runs/20260730-review-baselines-test-s0-a1/`
- Active experiment board: `.agents/REVIEW_REVISION_TODO.md`
- Canonical project evidence language: `.agents/ACADEMIC.md`
