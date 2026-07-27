# MetricGAN+ Teacher–Student Speech Enhancement

Acesta este repository-ul canonic pentru experimentele **MetricGAN+ teacher–student**.
Codul executabil existent se află în [`code_and_documentation/`](code_and_documentation/).

Guvernanța proiectului, arhitectura curentă, contractul de date, registrul de
experimente, testele și indexul documentației pornesc din
[`AGENTS.md`](AGENTS.md) și [`.agents/INDEX.md`](.agents/INDEX.md).

## Limita proiectului

În acest repository intră:

- codul MetricGAN+ teacher–student;
- configurațiile rezolvate și comenzile exacte ale rulărilor;
- metricile, logurile, rapoartele și predicțiile agregate;
- checkpoint-urile teacher/student și modelele exportate;
- manifestele de proveniență și hash-urile artefactelor externe.

Nu intră:

- codul, rulările sau ponderile MP-SENet;
- datasetul și copii ale fișierelor audio;
- `teacher_cache`/audio cache regenerabil;
- medii virtuale, cache-uri Python și fișiere temporare.

MP-SENet rămâne un proiect separat. Artefactele istorice nu se șterg și nu se
rescriu; sunt importate doar prin copiere verificată.

## Direcția canonică

Antrenarea nouă folosește exclusiv datasetul extern, read-only,
**VoiceBank+DEMAND**:

- teacher MetricGAN+ WB la 16 kHz;
- student causal WB la 16 kHz;
- student causal NB la 8 kHz;
- evaluare WB cu referință WB și PESQ-WB;
- evaluare NB cu referință NB și PESQ-NB.

Prima etapă folosește checkpoint-ul oficial
`speechbrain/metricgan-plus-voicebank`, fixat prin revizie și SHA-256, pentru a
antrena studenții S0-WB și S0-NB. A doua etapă îmbunătățește teacher-ul pornind
din același checkpoint, îl promovează numai după un câștig PESQ-WB real pe
`val_select` și guardrails STOI/SI-SDR, apoi antrenează de la zero studenții
S1-WB și S1-NB. Comparațiile S1–S0 izolează astfel schimbarea teacher-ului.

Planul verificabil este în
[`research_plan_voicebank_wb_nb.yaml`](code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml).
Poate fi auditat fără dataset:

```bash
SHARED_PYTHON=/path/to/shared-venv/bin/python
"$SHARED_PYTHON" scripts/validate_research_plan.py
```

Configurația portabilă este
[`configs/voicebank_campaign.yaml`](configs/voicebank_campaign.yaml). Copiile
locale ale manifestelor primesc rădăcina audio prin remapare și rămân în
`local/`, care nu este publicat. Datasetul nu este copiat, rescris sau inclus în
Git.

`campaign.py` validează că rădăcina rulărilor este în afara intrărilor și că
manifestele aparțin exclusiv VoiceBank+DEMAND. Comenzile de antrenare/cache
refuză CPU și un mediu diferit de venv-ul shared; validarea read-only, testele
și auditurile pot rula pe CPU.

Output-urile teacher-ului sunt memorate într-un cache persistent, ignorat de
Git și aflat local pe Desktop. Cache-ul este identificat prin hash-urile
checkpoint-ului, manifestului și contractului de stocare; etichetele etapelor
nu dublează același conținut. Păstrează în FP16 numai waveform-ul teacher și
masca ERB și nu copiază fișierele noisy/clean din dataset.

Controlerul canonic este `campaign.py`:

```bash
export METRICGAN_MANIFEST_ROOT=/path/to/local/manifests/voicebank_v1/full
export METRICGAN_RUN_ROOT=/path/to/repository/local/runs
export METRICGAN_SHARED_VENV=/path/to/shared-venv

"$METRICGAN_SHARED_VENV/bin/python" campaign.py validate
"$METRICGAN_SHARED_VENV/bin/python" campaign.py pilot-all --run-id <pilot-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py run-all --run-id <immutable-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py monitor-run --run-dir <run-dir>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py audit-run --run-dir <run-dir>
```

`pilot-all` și `run-all` refuză un worktree murdar. `smoke-all` poate fi folosit
numai pentru verificare tehnică, cu `--allow-dirty-smoke`. Smoke-ul și pilotul
nu produc rezultate promovabile.

## Discriminatorul metric și extensia TTS

PESQ nu este folosit direct ca o funcție diferențiabilă. Un predictor PESQ
înghețat furnizează gradientul pentru generator. Scorul prezis este normalizat
la intervalul MetricGAN `[0, 1]` și optimizat prin MSE către ținta curată `1`;
T1 este ancorat și la output-ul checkpoint-ului oficial T0 din cache-ul local.
În campania canonică actuală, ablation-ul metric este aplicat teacher-ului WB.
Studenții S0 și S1 folosesc același obiectiv de distilare `D1`, aceeași
arhitectură și același schedule; singura variabilă intenționată este
teacher-ul. Un viitor ablation metric direct pe studenți trebuie declarat
separat și ar necesita proxy-uri WB/NB distincte.

Interfața `MetricGANGeneratorObjective` poate fi conectată ulterior la un
generator TTS. Aceasta este momentan o direcție planificată: proxy-ul trebuie
recalibrat pe ieșirile modelului TTS înainte de a putea susține o concluzie
despre sinteză.

## Artefacte și GitHub

Modelele din `experiments/` sunt urmărite în Git normal; cel mai mare checkpoint
inventariat este sub limita GitHub per fișier. Cache-urile regenerabile și
manifestele foarte mari rămân externe, dar sunt reprezentate prin inventare cu
dimensiune, locație și hash unde este practic. Politica completă este în
[`docs/ARTIFACT_POLICY.md`](docs/ARTIFACT_POLICY.md), iar auditul inițial este
în [`docs/audits/2026-07-26-initial-audit.md`](docs/audits/2026-07-26-initial-audit.md).
