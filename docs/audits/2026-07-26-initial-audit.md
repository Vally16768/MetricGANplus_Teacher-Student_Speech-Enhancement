# Audit inițial — 2026-07-26

## Concluzie observată

Repository-ul public a fost clonat curat la commit
`5129bae87419d6879c4f15f5c464cd312faba281`.

Pe Kingston există trei surse distincte:

1. proiectul curent fără `.git`:
   logical source `legacy-worktree`;
2. un repository istoric cu modificări necomise:
   logical source `legacy-results-copy`;
3. rezultatele mixte:
   logical source `legacy-metricgan-results`.

Repository-ul istoric este la commit `2cabc05`, pe branch `master`, și are atât
fișiere modificate, cât și fișiere neversionate. El nu este considerat un
snapshot istoric exact până când provenance-ul fiecărei rulări este verificat.

## Matrice de arhitectură

| Familie | Evidență curentă | Status în acest repo |
|---|---|---|
| MetricGAN+ teacher mare | cod/config/checkpoint public | inclus |
| MetricGAN+ student causal S/XS/N6 | cod/config/checkpoint public | inclus |
| MetricGAN+ student QAT | cod/config/checkpoint public | inclus |
| MP-SENet teacher/student | directoare și cod separat pe Kingston | exclus și protejat |
| FullSubNet+/CMGAN teacher recovery | config/rulări istorice | alternativă istorică, nu pipeline canonic |

## Inventar inițial al experimentelor

Sub `results/metricgan` au fost observate 48 de directoare directe:

- 23 clasificate ca MP-SENet și excluse;
- 23 clasificate ca MetricGAN candidate;
- 2 clasificate ca teacher alternativ FullSubNet/CMGAN și excluse din pipeline-ul
  canonic.

Clasificarea nominală este doar un filtru inițial. Importul verifică și fiecare
cale internă.

O rulare importantă,
`teacher_recovery_metricgan_refiner_voicebank_preserve_20260406`, ocupă
aproximativ 163,60 GB. Aproximativ 163,56 GB reprezintă `outputs/teacher_cache`;
checkpoint-urile acelei rulări ocupă aproximativ 9,7 MB. Cache-ul nu este o
pondere de model și este regenerabil.

Repository-ul istoric ocupă aproximativ 1,3 GB, dintre care aproximativ 907 MB
sunt `outputs/`. Aproximativ 873 MB sunt CSV-uri, majoritatea manifeste/copy
plans; ponderile `.pt` din acel output ocupă aproximativ 30 MB.

Inventarul complet al ponderilor MetricGAN candidate a găsit:

- 1.665 referințe la fișiere de model/state;
- aproximativ 1,87 GB nominal;
- 968 obiecte SHA-256 unice;
- aproximativ 1,19 GB după deduplicare după conținut.

Importul verificat pe Desktop conține:

- 2.722 fișiere din `results/metricgan`;
- 175 fișiere din worktree-ul istoric;
- zero hash-uri greșite;
- zero fișiere ilizibile în selecția importată;
- zero căi MP-SENet, FullSubNet sau CMGAN.

Inventarul machine-readable este în
[`kingston-results-inventory.json`](kingston-results-inventory.json), iar fiecare
import conține propriul `import_manifest.json`.

## Matrice de validitate a protocolului

| Verificare | Stare |
|---|---|
| Datasetul nu este copiat/modificat | verificat pentru smoke `prepare_data`; hash-urile celor 4 manifeste au rămas identice |
| Split train/val/test fără overlap | verificat la nivel de pair/clean key pentru manifestele runtime curente |
| Commit exact pentru fiecare rulare | nerezolvat |
| Config rezolvat pentru fiecare rulare | parțial |
| Checkpoint ancestry | nerezolvat |
| Separare MP-SENet / MetricGAN+ | implementată pentru importurile noi |
| Test hold-out folosit o singură dată | nerezolvat |
| Multi-seed comparabil | parțial; trebuie reconstruit din artefacte |

Smoke-ul entry point-ului real `repro.py prepare_data` a trecut pe configul local.
Au fost observate 463.771 perechi train, 4.224 val-rank, 38.395 val-select și
51.270 test, fără overlap de pair sau clean key între train/val/test. Fișierele
derivate au fost scrise numai în `local/runs/smoke_local_setup/` pe Desktop.

## Rezultate vechi → rezultate corectate

Nu s-a reclasificat și nu s-a corectat încă nicio valoare. Toate valorile
existente rămân `observed`. O valoare devine `reproduced` sau `corrected` numai
după o rerulare dintr-un commit verificat, cu config rezolvat și același split.

## Graful rerulărilor

```text
dataset extern + manifeste înghețate
                |
                v
       teacher MetricGAN+ (seed)
                |
                v
         teacher cache extern
                |
                v
      student stage1 (seed-uri declarate)
                |
                v
              QAT
                |
                v
 val_rank -> val_select -> test final
```

Planul de seed va fi fixat după inventarul configurațiilor istorice; nu se
lansează o matrice înainte de un smoke test al entry point-ului real.

## Schimbări necesare în documentație/articol

- separarea explicită dintre MetricGAN+ și MP-SENet;
- marcarea valorilor istorice ca `observed`, nu automat `reproduced`;
- tabel per run cu commit/config/seed/checkpoint/split;
- descrierea faptului că datasetul și teacher cache-ul nu sunt în Git;
- raportarea mediei, deviației standard și intervalului de încredere pentru
  rerulările multi-seed.

## Întrebări rămase din auditul istoric

- Ce rulări au fost lansate din cod necomis?
- Care checkpoint teacher a inițializat fiecare student?
- Care manifeste sunt identice și care au fost regenerate?
- Ce rezultate FullSubNet+/CMGAN trebuie păstrate doar ca istoric și care trebuie
  eliminate din narațiunea MetricGAN+?

Rezolvat ulterior: configurațiile `teacher_recovery_*` nu sunt direcția
canonică; noul studiu folosește perechea WB `T0`/`T0_PESQ`, apoi studenții WB/NB.
Ponderile selectate vor folosi Git normal, nu Git LFS.
