# Documentation index

Every canonical Markdown document must appear here with one owner and one
status. Historical experiment directories are evidence, not canonical docs.

| Path | Role | Status |
|---|---|---|
| `AGENTS.md` | agent entry point | canonical |
| `README.md` | public project entry point | canonical |
| `.agents/INDEX.md` | project control index | canonical |
| `.agents/ARCHITECTURE.md` | architecture source of truth | canonical |
| `.agents/DATA.md` | data contract | canonical |
| `.agents/EXPERIMENTS.md` | experiment lifecycle/registry | canonical |
| `.agents/VALIDATION.md` | validation gates | canonical |
| `.agents/DOCUMENTATION_INDEX.md` | this registry | canonical |
| `.agents/ACADEMIC.md` | publication contract | canonical |
| `.agents/DECISIONS.md` | decision log | canonical |
| `.agents/TODO.md` | ordered execution and gate status | canonical |
| `.agents/EXECUTION_TODO.md` | detailed iterative P1–P6 execution board and evidence ledger | canonical active |
| `.agents/TEACHER_IMPROVEMENT_PLAN.md` | executed failed T1 calibration protocol and separate TTS-transfer boundary | canonical negative outcome |
| `docs/ARTIFACT_POLICY.md` | artifact policy | active, reconcile with this control plane |
| `docs/FINAL_RESULTS.md` | canonical article-facing S0 table, negative T1 outcome and claim-to-artifact map | canonical reproduced |
| `docs/audits/2026-07-26-initial-audit.md` | initial forensic audit | historical observed |
| `docs/audits/2026-07-26-cleanup-campaign-audit.md` | cleanup classification and dataset observation | active observed |
| `docs/audits/2026-07-26-legacy-surface-removal.md` | authorized legacy removal and recovery anchors | active observed |
| `docs/audits/2026-07-27-pilot-a1.md` | clean-snapshot pilot execution and scientific gate audit | active observed |
| `docs/audits/2026-07-27-full-a1-stopped.md` | controlled-stop audit and causal-max recovery evidence | active observed |
| `docs/audits/2026-07-27-official-teacher-baseline.md` | official checkpoint reconciliation and two-stage corrective design | active observed |
| `docs/audits/2026-07-27-two-stage-smoke-a3.md` | clean-snapshot seven-cell smoke, failed-gate fallback and cache-dedup audit | active observed |
| `docs/audits/2026-07-27-two-stage-pilot-a1.md` | monitored seven-cell pilot, proxy-exploitation evidence and failed teacher gate | active observed |
| `docs/audits/2026-07-27-bounded-teacher-smoke-a1.md` | bounded target-score/T0-anchor clean GPU smoke and independent audit | active observed |
| `docs/audits/2026-07-27-bounded-teacher-pilot-a1.md` | bounded frozen-proxy pilot, negative teacher gate and alternating-discriminator decision | active observed |
| `docs/audits/2026-07-27-alternating-teacher-smoke-a2.md` | corrected clean-target alternating D/G GPU smoke, local replay and failed verification gate | active observed |
| `docs/audits/2026-07-27-alternating-teacher-pilot-a1.md` | monitored alternating D/G pilot, current-output calibration failure and rejected T1 teacher | active observed |
| `docs/audits/2026-07-27-official-baseline-smoke-a1.md` | three-cell official T0→C0→S0 CUDA smoke, cache isolation and subset audit | active observed |
| `docs/audits/2026-07-27-official-baseline-full-a1.md` | audited full official baseline, bandwidth-matched metrics and 20-epoch convergence diagnosis | active observed |
| `docs/audits/2026-07-27-converged-s0-baseline-v1.md` | original converged S0 promotion audit; metrics superseded by true-length v2 | historical superseded |
| `docs/audits/2026-07-27-converged-s0-baseline-v2.md` | corrective true-length evaluation and canonical S0 promotion audit | active reproduced |
| `docs/audits/2026-07-27-teacher-calibration-t1-negative.md` | final calibrated-discriminator failure, stop decision and downstream non-execution | active reproduced negative evidence |
| `experiments/README.md` | experiment directory explanation | active, consolidate later |
| `experiments/runs/20260727-converged-s0-baseline-v1/reports/report.md` | padding-sensitive historical metric table | immutable superseded evidence |
| `experiments/runs/20260727-converged-s0-baseline-v2/reports/report.md` | canonical true-length S0 metric table and correction statement | immutable reproduced evidence |
| `code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml` | machine-independent canonical WB/NB campaign contract | canonical configuration, non-Markdown |

Rules:

- Update this table in the same change as any document add/remove/rename.
- A canonical fact must have one source of truth; other documents link to it.
- Mark old claims historical rather than silently rewriting their provenance.
- Remove deprecated documents only after useful unique content is migrated and
  a recoverable reference exists.
- Article prose is generated from promoted experiment records and canonical
  registers, never copied blindly from historical READMEs.
