# Addressed review — IEEE draft v2

Status: **review evidence complete; retained limitations explicit; manuscript editing remains manual**
Draft reviewed: `MetricGAN_Teacher_Student_IEEE_Draft_v2.pdf` (6 pages,
created 2026-07-30)

This document distinguishes completed evidence from planned work. A planned
experiment is not an addressed criticism.

| Reviewer concern | Current assessment | Action/evidence | Article change | Status |
|---|---|---|---|---|
| Distillation benefit not isolated | Valid major concern | Frozen WB loss-component matrix and utterance-paired intervals validated | Add the audited ablation table and restrict causal attribution to measured contrasts | passed |
| NB teacher has privileged >4-kHz information | Valid information mismatch | Matched-input teacher and clean-only NB student completed on 824 pairs; paired matrix audited | Rewrite target-generation text and limitations | passed with caveat |
| Reported 16-ms lookahead | Incorrect mathematical dependency | WB/NB future-perturbation regression validates the 160/80-sample (10-ms) bound; stateful streaming and device latency remain unmeasured | Replace all 16-ms claims and Table V entries with the validated wording | passed |
| NB results lack baselines | Valid major concern | Current-protocol noisy NB, clean-only NB, complete NB and matched-input teacher reference are validated | Expand current-results table within NB protocol | passed |
| One seed | Valid major concern retained by scope decision | Additional complete-model seeds were explicitly removed; 824-pair utterance bootstrap is complete but does not estimate training variability | State that training variability is unknown; do not report across-seed mean or standard deviation | open limitation |
| Reproducibility identifiers absent | Evidence exists but is missing from draft | Publish sanitized checkpoint, code/config, manifest and speaker-split hashes | Add reproducibility subsection/table | passed |
| Foundational references absent | Valid presentation/scholarship concern | Add original MetricGAN, MetricGAN+, VoiceBank+DEMAND, PESQ, STOI, SI-SDR and KD references | Replace the artificial 2024--2026 restriction | passed |
| Same FFT sizes at 16/8 kHz | Valid design asymmetry | State exact time supports; duration-matched-resolution retraining remains outside scope | Label as limitation; do not claim temporal equivalence | passed with limitation |
| “Energy” in Eq. (2) | Terminology is incorrect for summed magnitude | Rename to ERB-band weighted magnitude without changing the trained objective | Correct equation prose and symbols | passed |
| Complexity incomplete | Valid | Teacher/student parameter, weight-storage, neural-core MAC/s and recurrent-state-only counts reconciled; whole-waveform spectral tensors, activations and target-device costs remain explicitly unmeasured | Replace Table V and retain the non-benchmark caveat | passed |
| No subjective/artifact evaluation | Not currently measured | Provide sanitized audio/examples only if publication packaging permits; do not invent listening evidence | Keep as explicit limitation/future work | open limitation |
| Figure/visual density | Valid | Training/inference diagram and separate audited WB/NB plots prepared | Revise figures and shorten abstract | passed |
| Historical/refinement discussion | Too long and weakly evidenced in draft | Retain only concise, traceable negative-result statement or remove | Shorten Sections V-A/V-C | passed |
| Historical SI-SDR discrepancy | Protocols differ and need explicit causal explanation | Identify differences in teacher, architecture, segmentation and evaluation; never pool values | Add a short non-comparability note or remove historical table | passed with caveat |

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
| Fixed official T0 teacher | WB, 16 kHz, PESQ-WB | 824 | 3.1309 | 0.9319 | 8.5791 |
| A-COMPLETE, seed 0 (promoted reuse) | WB, 16 kHz, PESQ-WB | 824 | 3.0519 | 0.9296 | 9.0499 |
| Noisy input | NB, 8 kHz, PESQ-NB | 824 | 2.9481 | 0.9212 | 8.4318 |
| Fixed WB teacher with matched 8-kHz-bandlimited input | NB, 8 kHz, PESQ-NB | 824 | 3.6058 | 0.9283 | 9.2583 |
| N-CLEAN, seed 0 | NB, 8 kHz, PESQ-NB | 824 | 3.3445 | 0.9347 | 18.0940 |
| N-COMPLETE, seed 0 (promoted reuse) | NB, 8 kHz, PESQ-NB | 824 | 3.6151 | 0.9294 | 9.0709 |

These rows must not be compared across PESQ modes. The matched-input teacher
removes access to frequencies above 4 kHz at its input, but it is not a
dedicated narrowband model. N-CLEAN supplies the missing student reference.
Its much higher SI-SDR and lower PESQ than N-COMPLETE are aggregate
observations. The completed paired uncertainty audit supports only the
fixed-checkpoint trade-off wording below, not across-seed causal language.

## Completed ablation evidence and paired uncertainty

The completed WB seed-0 review cells below used the same frozen splits,
architecture, optimizer policy and checkpoint-selection protocol. A-CLEAN
completed 50 epochs and selected epoch 45; A-TWAVE early-stopped at epoch 40
and selected epoch 32; A-MASK early-stopped at epoch 39 and selected epoch 31;
A-TWAVE-MASK early-stopped at epoch 42 and selected epoch 34. The untouched
824-pair test was evaluated only after selection in each run.

| Cell | Protocol | Pairs | PESQ | STOI | SI-SDR (dB) | Delta SNR (dB) |
|---|---|---:|---:|---:|---:|---:|
| A-CLEAN, seed 0 | WB, 16 kHz, PESQ-WB | 824 | 2.5517 | 0.9350 | 17.9958 | 9.6458 |
| A-TWAVE, seed 0 | WB, 16 kHz, PESQ-WB | 824 | 3.0389 | 0.9278 | 8.7718 | -0.4849 |
| A-MASK, seed 0 | WB, 16 kHz, PESQ-WB | 824 | 3.0529 | 0.9288 | 8.6783 | -0.4990 |
| A-TWAVE-MASK, seed 0 | WB, 16 kHz, PESQ-WB | 824 | 3.0565 | 0.9292 | 8.7173 | -0.4631 |
| A-COMPLETE, seed 0 (promoted reuse) | WB, 16 kHz, PESQ-WB | 824 | 3.0519 | 0.9296 | 9.0499 | -0.1566 |

The result-summary hash, restricted checkpoint metadata and aggregate metric
support reconcile, and the repository validation suite passes. The complete
D1 WB row is reused from the promoted seed-0 package.

The first N-CLEAN run, `20260801-review-clean-nb-s0-a1`, failed at epoch and
global step `0/0`, before any optimizer update. Its 16-kHz source lengths were
mixed with 8-kHz crop coordinates, producing unequal batch tensors during
default collation. The failed directory is preserved as implementation
evidence and contains no scientific result. A focused target-rate loader fix
now passes short/long NB, default-collation, WB-preservation and real-manifest
batch checks. Clean commit `f69da47` supplies the required new snapshot:
smoke `20260801-review-clean-nb-smoke-s0-a2` passed without reading test. Full
run `20260801-review-clean-nb-s0-a2` then selected epoch 35 and early-stopped
at epoch 43. Its 824-pair NB test produced PESQ 3.344523, STOI 0.934739,
SI-SDR 18.094036 dB and delta-SNR 9.752861 dB. The test began only after
selection; the command/config, 43-row history, manifests, result/checkpoint
hashes and restricted checkpoint load independently reconcile. Reporting-only
run `20260801-review-matrix-uncertainty-s0-a1` regenerated all sample-level
metrics and passed independent reconciliation.

### Audited paired contrasts

All intervals below are deterministic 10,000-draw percentile bootstrap
intervals over the same 824 test utterances. Delta is left minus right, so a
positive value favors the left system for the displayed metric. WB and NB are
never mixed. These intervals quantify utterance-level uncertainty for the
fixed seed-0 checkpoints; they do not quantify training-seed variability.

| Contrast | Metric | Mean delta | Paired 95% CI |
|---|---|---:|---:|
| A-TWAVE vs A-CLEAN | PESQ-WB | +0.4872 | [+0.4673, +0.5072] |
| A-MASK vs A-CLEAN | PESQ-WB | +0.5012 | [+0.4813, +0.5208] |
| A-TWAVE-MASK vs A-CLEAN | PESQ-WB | +0.5048 | [+0.4852, +0.5240] |
| A-COMPLETE vs A-CLEAN | PESQ-WB | +0.5002 | [+0.4808, +0.5194] |
| A-TWAVE-MASK vs A-TWAVE | PESQ-WB | +0.0176 | [+0.0115, +0.0239] |
| A-TWAVE-MASK vs A-MASK | PESQ-WB | +0.0036 | [-0.0019, +0.0095] |
| A-COMPLETE vs A-TWAVE-MASK | PESQ-WB | -0.0045 | [-0.0083, -0.0008] |
| A-COMPLETE vs A-TWAVE-MASK | SI-SDR (dB) | +0.3326 | [+0.3051, +0.3610] |
| N-COMPLETE vs N-CLEAN | PESQ-NB | +0.2706 | [+0.2576, +0.2836] |
| N-COMPLETE vs N-CLEAN | STOI | -0.0053 | [-0.0062, -0.0045] |
| N-COMPLETE vs N-CLEAN | SI-SDR (dB) | -9.0231 | [-9.2528, -8.7927] |
| N-CLEAN vs noisy NB | PESQ-NB | +0.3964 | [+0.3773, +0.4157] |
| N-CLEAN vs noisy NB | SI-SDR (dB) | +9.6622 | [+9.3756, +9.9480] |

The evidence supports a trade-off statement, not a universal superiority
claim. Teacher-target components raise PESQ substantially relative to
clean-only training, while clean-only training retains much higher STOI and
SI-SDR. The combined teacher waveform and mask improves PESQ over waveform
alone, but its PESQ difference from mask alone has an interval spanning zero.
Adding the clean component to the combined WB teacher targets slightly lowers
PESQ while improving SI-SDR. The NB complete model shows the same PESQ versus
signal-fidelity trade-off relative to N-CLEAN.

## Article-ready corrections supported without new training

### Reproducibility identifiers

The revised reproducibility subsection should report the following stable
identities. The final publication-code commit will be updated once the review
campaign closes.

| Item | Identifier |
|---|---|
| Canonical seed-0 package code commit | `1770aa636f80042274e3181ae35f3e95f6aeb838` |
| SpeechBrain MetricGAN+ generator revision | `a196ce26b3bdace6fa1d819017584bdbcce462a8` |
| Raw official generator checkpoint SHA-256 | `147bfb866bac8264603546e035bf283370e716ed2f4b7412d308d2bcee88304f` |
| Portable T0 checkpoint SHA-256 | `5ece6fbd1ac16cca6df11ea724fb5e3710d6611049f54bbc8d126c79dbbc65d8` |
| Seed-0 WB student SHA-256 | `dc1d2d2171876fb5665bd447506e3371492a4619cc8f2749cbfca7292f1ca335` |
| Seed-0 NB student SHA-256 | `1b89e6b5931eb3a4bb63db7844ffe5e74486e9bf75b835342926776336d11491` |
| Resolved config SHA-256 | `d6989c259c51a43b75f66602e94964657405c21520f34c3a8b5dcc3d28ec9690` |
| `train_fit` manifest SHA-256 | `5715037a664ecb6a7302393a8b973e8744261d95e9487af17c397cdd65642bf6` |
| `val_rank` manifest SHA-256 | `5d3df5fa11ca64186117620d07f0beeeac0563ba93cffc85bcba48f38ad62a9e` |
| `val_select` manifest SHA-256 | `328e86032d166dc68d01482658b4073f25624ba0d88071c1862c3e0e1ea68d23` |
| Test manifest SHA-256 | `b74a2d9002d1a97a0c7f92d483d9c8840e2898d894222323d47f0aa630684e1d` |
| Validation speakers excluded from optimization | `p239`, `p244`, `p270`, `p286` |

The teacher implementation parity record uses SpeechBrain `v1.1.0`, source
revision `36c180c7bfad3bf5c48bd76a24799812952c4565`.

### Corrected lookahead wording

Validated replacement text:

> The recurrent mapping is frame-causal. Each student uses a centered 20-ms
> Hamming analysis window, so the mathematical future signal dependency is
> 10 ms (160 samples at 16 kHz or 80 samples at 8 kHz). The larger
> `n_fft/2` padding used by the library is not additional nonzero signal
> support because the analysis window is zero-padded inside the FFT buffer.
> This quantity is algorithmic analysis lookahead, not measured end-to-end
> latency; buffering, feature computation, recurrent execution, synthesis and
> device I/O remain to be benchmarked.

The revised paper must replace `16 ms` in the abstract, Section III-C,
Section VI-A, Table V and the conclusion. The numerical regression perturbs
only input samples at and after a hop-aligned cutoff. For both canonical
profiles, every output sample before `cutoff - win_length/2` remains
bit-identical, while the immediately following 10-ms boundary region changes:
160 samples at 16 kHz and 80 samples at 8 kHz. This validates the algorithmic
future-dependency bound of the offline operator. The current implementation
still processes whole waveforms with centered library STFT/iSTFT calls; it has
no audited stateful chunk API, hop-by-hop equivalence result or end-to-end
latency benchmark, and the paper must not imply any of those measurements.

### Loss terminology and cross-band time supports

Equation (2) computes a weighted sum of spectral magnitudes, not energy. Rename
`E_u(b,t)` to an **ERB-band weighted magnitude** (or equivalent wording)
without changing the already trained objective. Do not square the magnitude
retroactively, because that would define a different method requiring new
training.

The multi-resolution FFT sizes `{256, 512, 1024}` span `{16, 32, 64}` ms at
16 kHz and `{32, 64, 128}` ms at 8 kHz. The revised paper must state this
asymmetry explicitly and must not claim time-duration equivalence. A
duration-matched NB ablation is optional future evidence unless separately
predeclared and trained.

### Teacher-relative neural-core complexity

For the fixed teacher, the two bidirectional 200-unit LSTM layers and
`400x300`/`300x257` projections require approximately 1,888,300 neural-core
MACs per frame. At a 256-sample hop and 16 kHz (62.5 frames/s), this is
approximately 118.02 million MAC/s. The replacement table is:

| Model | Parameters | FP32 weights | INT8 weight-only lower bound | Frames/s | Neural-core MAC/s | Reduction from teacher |
|---|---:|---:|---:|---:|---:|---:|
| Fixed MetricGAN+ teacher | 1,895,514 | 7.582 MB | 1.896 MB | 62.5 | 118.02 million | - |
| WB student | 604,386 | 2.418 MB | 0.604 MB | 100 | 60.22 million | 49.0% |
| NB student | 514,018 | 2.056 MB | 0.514 MB | 100 | 51.21 million | 56.6% |

Storage uses decimal MB and counts weights only. The INT8 column is a
theoretical one-byte-per-weight lower bound, not an exported or accuracy-tested
integer model. The exact FP32 recurrent-state-only count for either student is
`3 layers x 160 hidden values x 4 bytes = 1.920 kB`. The teacher's LSTM state
alone is `2 layers x 2 directions x (hidden + cell) x 200 x 4 bytes = 6.400
kB`; because the teacher is bidirectional, this does not make its inference
streamable and excludes the full reverse-time input/activation requirement.

The arithmetic counts cover recurrent and linear MACs only. The current
PyTorch inference path processes whole waveforms and materializes centered
STFT/iSTFT spectra, magnitudes, masks and intermediate activations whose peak
memory scales with batch size and utterance length. No stateful streaming
kernel, fixed backend buffer plan, FFT cost/workspace, nonlinear and
element-wise cost, memory traffic, framework overhead, worst-case hop runtime,
end-to-end latency or energy has been measured. Table V must therefore label
the reported values as analytical architectural indicators, not processor
benchmarks or deployment results.

### Foundational references to add

- S.-W. Fu, C.-F. Liao, Y. Tsao, and S.-D. Lin, “MetricGAN: Generative
  Adversarial Networks based Black-box Metric Scores Optimization for Speech
  Enhancement,” *Proceedings of Machine Learning Research*, vol. 97,
  pp. 2031-2041, 2019.
- S.-W. Fu et al., “MetricGAN+: An Improved Version of MetricGAN for Speech
  Enhancement,” *Interspeech*, 2021, doi:
  `10.21437/Interspeech.2021-599`.
- C. Valentini-Botinhao et al., “Investigating RNN-based Speech Enhancement
  Methods for Noise-Robust Text-to-Speech,” *SSW*, 2016, doi:
  `10.21437/SSW.2016-24`, together with the exact VoiceBank+DEMAND data
  release used by the experiment.
- A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra,
  “Perceptual Evaluation of Speech Quality (PESQ),” *ICASSP*, 2001, doi:
  `10.1109/ICASSP.2001.941023`, and ITU-T Recommendation P.862.
- C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An Algorithm for
  Intelligibility Prediction of Time-Frequency Weighted Noisy Speech,”
  *IEEE/ACM TASLP*, 2011, doi: `10.1109/TASL.2011.2114881`.
- J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “SDR - Half-baked or
  Well Done?” *ICASSP*, 2019, doi: `10.1109/ICASSP.2019.8683855`.
- G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural
  Network,” arXiv:`1503.02531`, 2015.

## Evidence locations

- Standalone Word response/evidence pack:
  `MetricGAN_IEEE_Review_Response_Evidence_Pack.docx` with repository-root
  usage guidance in `README_REVIEW.md`
- Canonical seed-0 package:
  `experiments/runs/20260727-converged-s0-baseline-v2/`
- Full current-protocol reference evidence:
  `local/runs/20260730-review-baselines-test-s0-a1/`
- Active experiment board: `.agents/REVIEW_REVISION_TODO.md`
- Canonical project evidence language: `.agents/ACADEMIC.md`
