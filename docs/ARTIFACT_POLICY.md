# Politica artefactelor

## Decizie

Păstrăm un singur repository canonic pentru MetricGAN+ teacher–student:
`Vally16768/MetricGANplus_Teacher-Student_Speech-Enhancement`.

MP-SENet teacher–student rămâne separat. Nu mutăm artefacte MP-SENet în acest
repository și nu suprascriem nicio rulare istorică.

## Clase de date

| Clasă | Locație canonică | Versionare |
|---|---|---|
| Cod, config sursă, documentație | Git | Git normal |
| Config rezolvat, comandă, environment lock | `experiments/runs/<run_id>/provenance/` | Git normal |
| Metrici, loguri, rapoarte, grafice | `experiments/runs/<run_id>/` | Git normal |
| Checkpoint-uri și modele exportate | `experiments/runs/<run_id>/models/` | Git normal |
| Dataset audio | Kingston, în afara repo-ului | read-only logic; manifest + hash |
| `teacher_cache`, staged audio, cache eval | în afara Git | regenerabil; inventar agregat |
| Artefact istoric prea mare | locația istorică până la migrarea într-un artifact store | index cu hash/dimensiune |

GitHub nu este backup pentru dataset sau pentru cache-ul audio. GitHub este
backup-ul reproductibil pentru cod, provenance, rezultate și ponderile utile.

## Contract pentru rulări noi

Fiecare rulare nouă primește un `run_id` unic și un director nou. Nu se
refolosește un director vechi.

Minimum obligatoriu:

1. `config_resolved.yaml`;
2. `command.txt`;
3. `provenance.json` cu commit, stare dirty, seed și hash-uri de manifest;
4. log complet;
5. metrici raw și agregate;
6. checkpoint selectat și, dacă există, ultimul state de resume;
7. o stare de validitate: `observed`, `reproduced`, `corrected` sau `unresolved`.

## Import istoric

Importul este numai prin copiere:

- sursa nu se modifică;
- destinația existentă nu se suprascrie;
- fișierul copiat este verificat SHA-256;
- căile cu `mpsenet`, `mp_senet`, `MP-SENet`, `teacher_cache`, audio sau cache
  regenerabil sunt excluse;
- excluderile sunt consemnate în manifest.

Datasetul rămâne pe Kingston. Codul validează că toate rădăcinile de scriere
sunt în afara rădăcinilor datasetului.

## Publicare

Înainte de commit/push:

1. se rulează scanarea de secrete și de căi private;
2. niciun fișier nu trebuie să depășească 100 MiB;
3. pachetul unui push trebuie să rămână sub limita GitHub de 2 GB;
4. se validează că nu există MP-SENet sau audio în staging;
5. se verifică din nou manifestul de import.
