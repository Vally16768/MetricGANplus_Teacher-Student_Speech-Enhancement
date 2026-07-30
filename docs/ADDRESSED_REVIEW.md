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

Provisional replacement text, pending the numerical dependency/streaming test:

> The recurrent mapping is frame-causal. Each student uses a centered 20-ms
> Hamming analysis window, so the mathematical future signal dependency is
> 10 ms (160 samples at 16 kHz or 80 samples at 8 kHz). The larger
> `n_fft/2` padding used by the library is not additional nonzero signal
> support because the analysis window is zero-padded inside the FFT buffer.
> This quantity is algorithmic analysis lookahead, not measured end-to-end
> latency; buffering, feature computation, recurrent execution, synthesis and
> device I/O remain to be benchmarked.

The revised paper must replace `16 ms` in the abstract, Section III-C,
Section VI-A, Table V and the conclusion. It must not claim hop-by-hop
equivalence until R4's numerical test passes.

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
approximately 118.02 million MAC/s.

| Model | Frames/s | Neural-core MAC/s | Reduction from teacher |
|---|---:|---:|---:|
| Fixed MetricGAN+ teacher | 62.5 | 118.02 million | - |
| WB student | 100 | 60.22 million | 49.0% |
| NB student | 100 | 51.21 million | 56.6% |

These counts cover recurrent and linear arithmetic only. The article must keep
the caveat that STFT/iSTFT, nonlinearities, memory traffic, recurrent state,
spectral buffers, framework overhead, wall-clock latency and energy are not
included or measured.

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

- Canonical seed-0 package:
  `experiments/runs/20260727-converged-s0-baseline-v2/`
- Full current-protocol reference evidence:
  `local/runs/20260730-review-baselines-test-s0-a1/`
- Active experiment board: `.agents/REVIEW_REVISION_TODO.md`
- Canonical project evidence language: `.agents/ACADEMIC.md`
