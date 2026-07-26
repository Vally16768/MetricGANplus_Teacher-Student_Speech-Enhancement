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
| `docs/ARTIFACT_POLICY.md` | artifact policy | active, reconcile with this control plane |
| `docs/audits/2026-07-26-initial-audit.md` | initial forensic audit | historical observed |
| `docs/audits/2026-07-26-cleanup-campaign-audit.md` | cleanup classification and dataset observation | active observed |
| `docs/audits/2026-07-26-legacy-surface-removal.md` | authorized legacy removal and recovery anchors | active observed |
| `experiments/README.md` | experiment directory explanation | active, consolidate later |
| `code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml` | machine-independent canonical WB/NB campaign contract | canonical configuration, non-Markdown |

Rules:

- Update this table in the same change as any document add/remove/rename.
- A canonical fact must have one source of truth; other documents link to it.
- Mark old claims historical rather than silently rewriting their provenance.
- Remove deprecated documents only after useful unique content is migrated and
  a recoverable reference exists.
- Article prose is generated from promoted experiment records and canonical
  registers, never copied blindly from historical READMEs.
