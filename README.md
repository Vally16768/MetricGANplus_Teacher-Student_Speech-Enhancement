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
"$METRICGAN_SHARED_VENV/bin/python" campaign.py run-baseline --run-id <baseline-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py continue-students \
  --source-run-dir <audited-baseline-run> --run-id <continuation-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py calibrate-teacher \
  --run-id <calibration-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py pilot-teacher \
  --calibration-run-dir <passed-calibration-run> --run-id <teacher-pilot-id>

# T3 direct-perceptual teacher pilot (requires passed T3 support)
"$METRICGAN_SHARED_VENV/bin/python" campaign.py train-t3-teacher \
  --support-run-dir <passed-t3-support-run> \
  --teacher-checkpoint <official-t0-checkpoint> \
  --teacher-cache-manifest <local-t0-wb-cache-manifest> \
  --run-id <t3-matched-pilot-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py pilot-all --run-id <pilot-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py run-all --run-id <immutable-id>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py monitor-run --run-dir <run-dir>
"$METRICGAN_SHARED_VENV/bin/python" campaign.py audit-run --run-dir <run-dir>
```

`run-baseline` execută numai T0 oficial, cache-ul C0 și studenții S0-WB/S0-NB,
apoi se oprește cu raport și audit. Nu construiește proxy și nu pornește
T1/S1. `pilot-all`, `run-all`, `pilot-baseline` și `run-baseline` refuză un
worktree murdar. `smoke-all` și `smoke-baseline` pot fi folosite numai pentru
verificare tehnică, cu `--allow-dirty-smoke`. Smoke-ul și pilotul nu produc
rezultate promovabile.

Rulările full ale studenților au un plafon de 50 de epoci, reduc LR la platou
și aplică early stopping după opt evaluări fără progres. `continue-students`
este folosit numai pentru un baseline full auditat care s-a oprit la un plafon
mai mic; creează un director nou și restaurează starea completă fără să
suprascrie rularea sursă.

`calibrate-teacher` ține E0 înghețat, actualizează numai discriminatorul și
aplică același gate strict după cel mult două refresh-uri, fiecare cu 100 de
output-uri curente held-out. `pilot-teacher` pornește
numai după un gate de calibrare trecut și execută exclusiv E0/E1/E2; un gate D
eșuat sare peste update-ul generatorului. Aceste comenzi nu creează C1 și nu
antrenează studenți.

## Baseline S0 publicat

Pachetul canonic este
[`20260727-converged-s0-baseline-v2`](experiments/runs/20260727-converged-s0-baseline-v2/).
Studentul WB a selectat epoca 34 și s-a oprit la 42, iar studentul NB a
selectat epoca 41 și s-a oprit la 49. Pe test, rezultatele sunt PESQ-WB
`3.051914` pentru S0-WB și PESQ-NB `3.615133` pentru S0-NB; cele două valori
aparțin unor protocoale diferite și nu se compară direct.

În timpul pregătirii T1 s-a constatat că evaluarea teacher-ului BLSTM din v1
era sensibilă la padding și `eval_batch_size`. V2 folosește inferență
per-utterance la lungimea reală pentru toate cele trei modele. Modelele,
hash-urile și selecția au rămas identice; v1 este păstrat numai ca dovadă
istorică supersedată și scorurile sale nu trebuie citate.

Pachetul public conține cele trei modele selectate, metrici, istoricele de
training, grafice, config portabil, proveniență și hash-uri. Nu conține
VoiceBank+DEMAND, audio generat, cache teacher, replay sau training state.
Raportul complet este
[`reports/report.md`](experiments/runs/20260727-converged-s0-baseline-v2/reports/report.md).

## Rezultatul fazei T1

Diagnosticul strict al discriminatorului pe output-urile curente a eșuat după
cele două refresh-uri predeclarate. Gate-ul final a avut MAE PESQ normalizat
`0.2133`, Pearson `0.5545` și Spearman `0.5435`, față de limitele
`0.06/0.80/0.80`. Generatorul nu a primit niciun update; E1/E2, C1 și
studenții S1 nu au fost porniți. Raportul negativ auditabil este
[`2026-07-27-teacher-calibration-t1-negative.md`](docs/audits/2026-07-27-teacher-calibration-t1-negative.md).

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
