# Documentatie Completa Pentru Experimentul Academic `VoiceBank+DEMAND + DNS5 -> MetricGAN Teacher -> Causal Student`

Acest document descrie complet experimentul academic curent din proiectul `MetricGANplus_Teacher-Student_Speech-Enhancement`.

Scopul lui este dublu:

1. sa fixeze protocolul stiintific corect al experimentului combinat `VoiceBank+DEMAND + DNS5`;
2. sa ofere un ghid operational suficient de detaliat pentru control, monitorizare, audit si reproductibilitate.

Documentul acopera:

- dataset-ul si split-urile efective;
- garantiile anti-leakage;
- modelele si arhitecturile;
- antrenarea teacher-ului si a student-ului;
- construirea teacher cache;
- metricele si implementarea lor;
- artefactele de tracking si fisierele care trebuie urmarite;
- criteriile de selectie si pragurile tinta;
- limitele actuale ale protocolului;
- punctele de inovatie ale directiei curente.

## 1. Obiectivul experimentului

Experimentul curent nu este linia clasica `8 kHz` documentata in `README.md` si `README_extended.md`.

Aici obiectivul este altul:

- se folosesc toate datele de antrenare disponibile in workspace-ul curent: `VoiceBank+DEMAND` si `DNS5`;
- se mentine un singur teacher de tip `metricgan_plus_native8k`, varianta `small`;
- teacher-ul este selectat dupa `VoiceBank val_select PESQ`, nu dupa un scor mixt;
- `DNS5` ramane in training si in validare ca semnal de robustete, nu ca selector principal;
- din teacher rezulta un student mic, cauzal, cu prag minim impus pe `VoiceBank val_select PESQ`.

Pragurile active din configul academic curat sunt:

- teacher: `VoiceBank val_select PESQ >= 3.10`
- student: `VoiceBank val_select PESQ >= 2.8605`

Configul academic curat folosit acum este `configs/scenario_combined_datasets_kingston_runtime_academic_clean_20260403_121456.yaml`.

## 2. Componentele principale ale sistemului

Fisierele cheie care definesc protocolul sunt:

- `repro.py`: orchestration CLI pentru `prepare_data`, `train_teacher`, `build_teacher_cache`, `train_stage1`, `train_qat`, `evaluate`, `run_all`
- `sebench/training.py`: bucla generica de training, selectie, early stopping, evaluare si persistenta de stare
- `sebench/models.py`: definitiile teacher/student si builder-ele de model
- `sebench/losses.py`: retetele de loss `T0`, `T0_PESQ`, `D1`, `D2` si modelul PESQ proxy
- `sebench/teacher_cache.py`: generarea teacher cache din teacher-ul FP32
- `metrics/pesq.py`, `metrics/stoi.py`, `metrics/sisdr.py`, `metrics/composite.py`, `metrics/snr.py`: metricele obiective
- `README_ACADEMIC_DATA_SPLITS.md`: politica de split, mai scurta si focalizata pe leakage
- `runbooks/kingston_runtime/README.md`: partea de bundle/runtime pe server

Punctele cele mai importante din cod pentru experimentul academic sunt:

- `repro.py:784`: pregatirea datasetului academic combinat
- `repro.py:1035`: construirea replay schedule-ului pentru DNS5
- `repro.py:1130`: override-urile de evaluare pentru teacher/student fara `test`
- `repro.py:1302`: antrenarea PESQ proxy-ului
- `repro.py:1590`: `command_prepare_data`
- `repro.py:1889`: `command_train_teacher`
- `repro.py:2162`: `command_train_stage1`
- `repro.py:2345`: `command_evaluate`
- `repro.py:2475`: `command_run_all`
- `sebench/training.py:935`: evaluarea unui manifest si agregarea metricalor
- `sebench/training.py:1480+`: logica de selectie, guardrail, early stopping, save/resume

## 3. Dataset-ul efectiv folosit

### 3.1 Sursele de date

Experimentul academic foloseste doua corpora:

- `VoiceBank+DEMAND` la `16 kHz`
- `DNS5 headset` la `16 kHz`

In config:

- `dataset.voicebank_root = /mnt/STORAGE/ulp-stack/data/voicebank-demand`
- `dataset.dns5_root = /mnt/STORAGE/ulp-stack/data/dns5-headset-16k`

### 3.2 Manifestele principale

Manifestele de baza pentru train/validation sunt deja materializate in runtime:

- `dataset.combined_train_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/train_combined_staged.csv`
- `dataset.combined_val_rank_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/val_rank_combined_staged.csv`
- `dataset.combined_val_select_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/val_select_combined_staged.csv`
- `dataset.combined_test_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/test_combined_staged.csv`

Manifestele per-domain folosite de protocolul academic sunt:

- `dataset.voicebank_train_fit_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/voicebank_train_fit.csv`
- `dataset.voicebank_val_rank_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/voicebank_val_rank.csv`
- `dataset.voicebank_val_select_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/voicebank_val_select.csv`
- `dataset.dns5_train_fit_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/dns5_train_fit.csv`
- `dataset.dns5_val_rank_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/dns5_val_rank.csv`
- `dataset.dns5_val_select_csv = /mnt/STORAGE/ulp-stack/data/manifests/metricgan/per_domain_runtime/dns5_val_select.csv`

### 3.3 Cardinalitati curente

Din summary-ul academic curat generat de `prepare_data`, cardinalitatile efective sunt:

| Split | VoiceBank | DNS5 | Combined |
|---|---:|---:|---:|
| `train_fit` | `9754` | `454017` | `463771` |
| `val_rank` | `128` | `4096` | `4224` |
| `val_select` | `1690` | `36705` | `38395` |
| `test` | `824` | `50446` | `51270` |

Sursa pentru aceste numere este:

- `outputs/prepare_data/academic_combined_explicit_summary.json`
- `outputs/combined/explicit_runtime_tests/*.csv`

In run-ul academic curent, summary-ul este la:

- `/mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs/prepare_data/academic_combined_explicit_summary.json`

### 3.4 Politica de split

`VoiceBank+DEMAND` foloseste:

- `train_fit` din train-ul corpusului
- `val_rank` si `val_select` din speaker holdout pe `p239`, `p286`, `p244`, `p270`
- `test` separat

`DNS5` foloseste:

- `train_fit`
- `val_rank`
- `val_select`
- `test`

Caveat important:

- in runtime-ul curent, `voicebank_test` si `dns5_test` sunt reconstruite din `combined_test_csv` daca manifestele brute separate nu sunt in bundle;
- `voicebank_test` este reconstruit din prefixul ordonat al `combined_test_csv`;
- `dns5_test` este reconstruit din sufixul ordonat al `combined_test_csv`.

Acest lucru este documentat explicit in `academic_combined_explicit_summary.json` la `notes`.

## 4. Garantiile anti-leakage

Protocolul academic curent a fost ajustat explicit pentru a elimina leakage-ul din training.

### 4.1 Ce NU se mai intampla

Teacher-ul si student-ul nu mai vad `test` in timpul trainingului.

Acest lucru este impus in doua locuri:

- `repro.py:1805` seteaza `test_csv = None` in configuratia de train
- `repro.py:1130` construieste pentru teacher/student doar seturile de validare:
  - `VoiceBank val_rank`
  - `VoiceBank val_select`
  - `DNS5 val_rank`
  - `DNS5 val_select`

`test` este folosit doar in `command_evaluate`, in `repro.py:2345`.

### 4.2 Auditul de integritate

La `prepare_data`, pipeline-ul face audit de integritate pe split-uri.

Summary-ul de integritate include:

- duplicate perechi `(noisy, clean)` per split
- duplicate `clean_key`
- overlap `train_vs_val`
- overlap `train_vs_test`
- overlap `val_vs_test`

Pentru run-ul academic curat, valorile sunt:

- `train_vs_val.clean_overlap = 0`
- `train_vs_test.clean_overlap = 0`
- `val_vs_test.clean_overlap = 0`
- `train_vs_val.pair_overlap = 0`
- `train_vs_test.pair_overlap = 0`
- `val_vs_test.pair_overlap = 0`

Acesta este criteriul minim pentru a considera protocolul curat metodologic.

### 4.3 Limita care ramane

Protocolul este curat din punctul de vedere al leakage-ului, dar nu este inca perfect comparabil cu o evaluare oficiala DNS Challenge:

- `DNS5 test` este un internal hold-out runtime, nu benchmark oficial extern.

## 5. Modelele folosite

### 5.1 Teacher

Teacher-ul academic curent este:

- family: `metricgan_plus_native8k`
- variant: `small`
- start checkpoint: `reference/checkpoints/metricgan_plus_native8k_small.pt`

Arhitectura teacher-ului in `sebench/models.py`:

- recurrent core: `2` straturi `bidirectional LSTM`
- `hidden_size = 200`
- `linear_dim = 300`
- input spectral based, cu predictie de masca

Teacher-ul este quality-first, non-causal, si serveste ca profesor pentru student.

### 5.2 Studentii candidati

Familiile student definite in config sunt:

- `metricgan_plus_native8k_causal_s`
- `metricgan_plus_native8k_causal_n6`

Configuratiile lor in `sebench/models.py` sunt:

| Familie | hidden_size | num_layers | linear_dim | rnn_type |
|---|---:|---:|---:|---|
| `metricgan_plus_native8k_causal_s` | `96` | `1` | `128` | `gru` |
| `metricgan_plus_native8k_causal_xs` | `64` | `1` | `96` | `gru` |
| `metricgan_plus_native8k_causal_n6` | `128` | `2` | `160` | `gru` |

In protocolul actual:

- candidatul principal este `metricgan_plus_native8k_causal_s`
- fallback-ul este `metricgan_plus_native8k_causal_n6`
- `causal_xs` exista in cod, dar nu este inclus in configul academic curent

### 5.3 Modelul PESQ proxy

`phase_c` foloseste un model auxiliar `PESQProxyRegressor` definit in `sebench/losses.py`.

Acesta:

- ia ca intrare `noisy`, `enhanced`, `clean`
- calculeaza reprezentari `log-magnitude STFT`
- construieste un stack de `6` canale de diferente/magnitudini
- trece prin `Conv2d -> GELU -> Conv2d -> GELU -> Conv2d -> GELU -> AdaptiveAvgPool2d`
- proiecteaza scorul intr-un scalar trecut prin `4.5 * sigmoid`

Rolul lui nu este sa inlocuiasca PESQ-ul real in raportare, ci sa introduca un semnal surrogate in scurtul finetune din `phase_c`.

## 6. Front-end-ul audio si setarile de baza

Configul academic curat are:

- `sample_rate = 16000`
- `n_fft = 512`
- `hop_length = 160`
- `win_length = 320`
- `segment_len = 32000`
- `erb_bands = 32`
- `context_frames = 5`

Aceste valori sunt folosite atat in teacher/student, cat si in PESQ proxy si teacher cache.

### 6.1 Setarile de runtime si optimizare

Setarile operationale importante din configul academic curat sunt:

- `training.batch_size = 8`
- `training.grad_accum = 1`
- `training.num_workers = 25`
- `training.prefetch_factor = 2`
- `training.persistent_workers = true`
- `training.pin_memory = true`
- `training.amp = true`
- `training.grad_clip = 5.0`
- `training.scheduler = plateau`
- `training.lr_factor = 0.5`
- `training.lr_patience = 3`
- `training.min_lr = 1e-6`
- `training.rank_eval_every = 1`
- `training.select_eval_every = 2`
- `training.checkpoint_every_minutes = 12`

Aceste setari explica direct comportamentul observat in runtime:

- de ce epocile fac save periodic chiar in interiorul antrenarii;
- de ce `val_rank` apare la fiecare epoca;
- de ce `val_select` ruleaza doar la fiecare 2 epoci;
- de ce initializarea este relativ usoara, fara `evaluate_init_checkpoint`.

## 7. Training flow complet

Flow-ul complet este definit in `repro.py:2475`, in `command_run_all`:

1. `prepare_data`
2. `train_teacher`
3. `build_teacher_cache`
4. `train_stage1`
5. optional `train_qat`
6. `evaluate`
7. optional `report`

In configul academic curent:

- `qat.auto_run = false`
- `report.enabled = false`

Asta inseamna ca flow-ul efectiv este:

1. `prepare_data`
2. `train_teacher`
3. `build_teacher_cache`
4. `train_stage1`
5. `evaluate`

Observatie operationala importanta:

- `command_train_stage1(...)` este idempotent din punct de vedere al dependintelor;
- daca lipseste `teacher_training summary`, relanseaza teacher-ul;
- daca lipseste teacher cache-ul, il construieste;
- deci `train_stage1` poate recupera singur o parte din flow daca este pornit separat.

## 8. Teacher training in detaliu

### 8.1 Pornirea teacher-ului

Teacher-ul porneste din `_resolve_teacher_start_checkpoint(...)`.

Ordinea de rezolvare este:

1. castigatorul existent dintr-un `teacher_training summary` deja prezent in output root
2. ultimul checkpoint gasit in run-urile teacher locale
3. `resume_checkpoint` din config, daca exista
4. `paths.teacher_source_checkpoint`

In configul academic curat:

- `resume_checkpoint = ''`
- `resume_training_state = ''`

Prin urmare, startul curent este checkpoint-ul de referinta `metricgan_plus_native8k_small.pt`.

### 8.2 Phase A: mixed continuation

`phase_a` este definita in config astfel:

- `lr = 5e-5`
- `epochs = 8`
- `early_stop_patience = 4`
- `min_epochs = 3`
- `loss_recipe = T0`

Date folosite:

- intregul `combined train_fit`

Scop:

- stabilizarea teacher-ului pe intreaga uniune `VoiceBank + DNS5`

### 8.3 Phase B: VoiceBank-biased replay adaptation

`phase_b` este definita astfel:

- `lr = 2e-5`
- `epochs = 10`
- `early_stop_patience = 4`
- `min_epochs = 4`
- `loss_recipe = T0`

Datele nu mai sunt luate ca raw union la fiecare epoca.

Se construieste un replay schedule prin `_build_replay_schedule_manifests(...)`:

- se citesc toate randurile `voicebank_train_fit`
- se citesc toate randurile `dns5_train_fit`
- se alege `shard_size = round(len(voicebank_train_fit) * dns_fraction)`
- pentru `phase_b`, `dns_fraction = 1.0`
- deci fiecare shard DNS5 are aproximativ dimensiunea train-ului VoiceBank
- fiecare epoca foloseste `VoiceBank train_fit` complet + un shard DNS5 rotativ

Consecinta:

- in fiecare epoca, VoiceBank este prezent integral
- DNS5 nu dispare, ci este parcurs integral de-a lungul fazei
- se evita dominarea bruta a optimization-ului de catre DNS5

### 8.4 Guardrail-ul DNS5

Dupa `phase_a`, teacher-ul calculeaza un prag minim pentru `DNS5 val_select PESQ`:

- `guardrail_floor = phase_a_dns5_val_select_pesq - 0.05`

In `phase_b` si `phase_c`, un checkpoint poate deveni `best` doar daca:

- imbunatateste selectorul principal `VoiceBank val_select PESQ`
- si trece guardrail-ul `DNS5 val_select PESQ >= guardrail_floor`

Aceasta regula este implementata prin:

- `selection_metric = val_select/pesq_mean`
- `selection_guardrail_metric = dns5_val_select/pesq_mean`
- `selection_guardrail_min = guardrail_floor`

### 8.5 Phase C: scurt PESQ-aware finetune

`phase_c` este definita astfel:

- `lr = 1e-5`
- `epochs = 4`
- `early_stop_patience = 2`
- `min_epochs = 2`
- `dns_fraction = 0.5`
- `loss_recipe = T0_PESQ`

Replay-ul DNS5 este mai usor decat in `phase_b`:

- `shard_size = round(len(voicebank_train_fit) * 0.5)`

Deci in aceasta faza accentul este si mai puternic pe VoiceBank, dar fara a scoate complet DNS5 din schema.

## 9. Loss-urile folosite

### 9.1 `T0`

Teacher-ul foloseste in `phase_a` si `phase_b` loss-ul `T0`:

- `T0 = 0.70 * spectral + 0.25 * wave + 0.05 * sisdr`

unde:

- `spectral` este `ComplexSTFTLoss`
- `wave` este `SmoothL1Loss(beta=0.5)`
- `sisdr` este `SISDRLoss`

### 9.2 `T0_PESQ`

`phase_c` foloseste:

- `t0_total = 0.70 * spectral + 0.25 * wave + 0.05 * sisdr`
- `pesq_proxy_loss = - predicted_pesq`
- `T0_PESQ = 0.60 * t0_total + 0.25 * pesq_proxy_loss + 0.15 * sisdr`

Observatie importanta:

- `phase_c` nu este MetricGAN+ adversarial canonic;
- este un finetune quality-aware cu surrogate de PESQ;
- trebuie documentat exact asa, nu ca reproducere fidela a trainer-ului original adversarial.

### 9.3 `D1`

Studentul foloseste `D1`:

- `D1 = 0.60 * teacher_mask + 0.25 * teacher_wave + 0.15 * spectral`

unde:

- `teacher_mask` compara masca ERB a studentului cu `teacher_mask_erb` din cache
- `teacher_wave` compara output-ul studentului cu `teacher_wav` din cache prin `ComplexSTFTLoss`
- `spectral` compara studentul cu `clean`

### 9.4 `D2`

QAT, daca este pornit, foloseste:

- `D2 = D1 + 0.05 * sisdr`

## 10. Construirea PESQ proxy-ului

PESQ proxy-ul este antrenat de `_train_teacher_pesq_proxy(...)`.

### 10.1 Datele sursa pentru proxy

Pentru fiecare domeniu se esantioneaza pana la:

- `max_samples_per_domain = 512`

Exemplele sunt generate din surse multiple:

- `noisy`
- `spectral_gating`
- `metricgan_raw` daca SpeechBrain este disponibil
- teacher checkpoint din `phase_a`
- teacher checkpoint din `phase_b`

Pentru fiecare exemplu se calculeaza PESQ real offline, iar record-ul stocat contine:

- `domain`
- `source`
- `noisy`
- `clean`
- `enhanced`
- `pesq`

### 10.2 Splitul proxy-ului

Setul proxy este amestecat cu seed fix, apoi impartit astfel:

- `10%` validation
- `90%` train

Configul curent pentru proxy:

- `epochs = 8`
- `batch_size = 8`
- `hidden_channels = 32`
- `projection_dim = 64`
- `lr = 1e-3`

Artefactele sunt scrise in:

- `outputs/teacher_pesq_proxy/proxy_records.json`
- `outputs/teacher_pesq_proxy/pesq_proxy.pt`
- `outputs/teacher_pesq_proxy/summary.json`

## 11. Teacher cache

Teacher cache-ul nu mai foloseste output-uri quantized pentru distilare.

Acesta este un punct critic al protocolului nou.

### 11.1 Ce contine

Teacher cache-ul este generat de `command_build_teacher_cache(...)` si `sebench/teacher_cache.py`.

Pentru fiecare exemplu din `combined train_fit`, cache-ul salveaza:

- `teacher_wav`
- `teacher_mask_erb`
- optional `guidance_sg`, daca ar fi activat ghidajul clasic

### 11.2 De ce este important

Distilarea trebuie sa invete din teacher-ul final FP32, nu dintr-o varianta cuantizata dinamic folosita doar pentru audit sau simulare.

Aceasta decizie elimina o sursa artificiala de degradare a studentului.

### 11.3 Setarile actuale

Configul teacher cache curent are:

- `batch_size = 64`
- `num_workers = 16`
- `pin_memory = true`
- `persistent_workers = true`
- `prefetch_factor = 2`

Implementarea:

- grupeaza micro-batch-urile pe lungimi compatibile
- ruleaza teacher-ul pe `device` cerut, inclusiv CUDA
- nu mai citeste audio `clean` inutil pentru cache build

Artefactele sunt in:

- `outputs/teacher_cache/teacher_cache.csv`
- `outputs/teacher_cache/summary.json`

## 12. Student training in detaliu

### 12.1 Ordinea candidatilor

Protocolul current este teacher-first, student-second.

Studentii sunt lansati astfel:

1. `metricgan_plus_native8k_causal_s`
2. `metricgan_plus_native8k_causal_n6` doar daca primul nu trece floor-ul

### 12.2 Phase S1

`phase_s1` foloseste:

- `lr = 5e-4`
- `epochs = 24`
- `early_stop_patience = 6`
- `min_epochs = 8`
- train pe `combined train_fit`
- loss `D1`

### 12.3 Phase S2

`phase_s2` foloseste:

- `lr = 2e-4`
- `epochs = 12`
- `early_stop_patience = 4`
- `min_epochs = 4`
- loss `D1`

Datele pentru `phase_s2` sunt construite exact ca la teacher `phase_b`:

- `VoiceBank train_fit` complet la fiecare epoca
- `DNS5` prin shard-uri replay rotative

Teacher cache-ul este si el filtrat pe fiecare manifest mixt al fazei, prin `_build_teacher_cache_schedule(...)`, astfel incat fiecare epoca sa consume doar perechile relevante din cache.

### 12.4 Regula de selectie a studentului

Regula de selectie este:

- se maximizeaza `VoiceBank val_select PESQ`
- student floor: `2.8605`
- daca familia principala este sub floor si fallback-ul este peste floor, se alege fallback-ul
- daca ambele depasesc floor-ul si diferenta este `<= 0.02 PESQ`, se prefera familia principala

Acest comportament este implementat in `_select_stage1_winner(...)`.

## 13. QAT

QAT este prezent in cod, dar este oprit in configul academic curent:

- `qat.auto_run = false`

Conditia de intrare in QAT este:

- studentul castigator trebuie sa fie deja peste floor-ul de `2.8605`

QAT foloseste:

- loss `D2`
- `lr = 2e-4`
- `epochs = 150`
- `early_stop_patience = 4`
- `min_epochs = 10`

In directia curenta, QAT nu face parte din flow-ul academic principal. Motivul este simplu:

- obiectivul imediat este optimizarea teacher-ului si a studentului academic corect;
- QAT poate fi readaugat ulterior, dupa ce linia teacher/student este stabila si competitiva.

## 14. Selectie, early stopping si praguri

### 14.1 Teacher

Teacher-ul este selectat dupa:

- `selection_metric = val_select/pesq_mean`

Aici `val_select` inseamna implicit `VoiceBank val_select`, deoarece acesta este manifestul principal pus in `val_select_csv`.

Guardrail optional:

- `selection_guardrail_metric = dns5_val_select/pesq_mean`
- `selection_guardrail_min = phase_a_dns5_val_select_pesq - 0.05`

Teacher target floor:

- `target_pesq_floor = 3.1`

### 14.2 Student

Student floor:

- `target_pesq_floor = 2.8605`

### 14.3 Evaluare la init

In configul academic curent:

- `teacher_training.evaluate_init_checkpoint = false`
- `stage1.evaluate_init_checkpoint = false`
- `qat.evaluate_init_checkpoint = false`

Aceasta decizie este operational importanta:

- evita evaluari initiale foarte scumpe pe zeci de mii de exemple inainte de `epoch 1`
- face pipeline-ul mai fluid
- nu schimba protocolul de selectie, doar elimina un overhead mare de startup

## 15. Metricile raportate

### 15.1 PESQ

Implementarea este in `metrics/pesq.py`.

Reguli:

- accepta doar `8000` sau `16000 Hz`
- foloseste `wb` la `16000 Hz`
- foloseste `nb` la `8000 Hz`
- daca backend-ul `pesq` arunca `NoUtterancesError` sau semnalul este invalid, scorul devine `NaN`

### 15.2 STOI

Implementarea este in `metrics/stoi.py`.

Scorul devine `NaN` daca rezultatul nu este finit.

### 15.3 SI-SDR

Implementarea este in `metrics/sisdr.py`.

Este o implementare stricta pe semnale 1D mono, cu verificare de lungime.

### 15.4 Delta SNR

`delta_snr` este folosita pentru raportare si este agregata in `evaluate_manifest(...)`.

### 15.5 Metricile composite

`CSIG`, `CBAK`, `COVL` sunt implementate in `metrics/composite.py`, pe baza formulelor Hu & Loizou.

Acestea sunt calculate doar cand `compute_composite = true`.

In configul academic curent:

- `rank_compute_composite = false`
- `select_compute_composite = true`

Deci:

- `val_rank` este mai ieftin si nu calculeaza composite metrics
- `val_select` si `test` calculeaza `CSIG`, `CBAK`, `COVL`

### 15.6 DNSMOS

Suportul exista, dar in experimentul academic curent este oprit:

- `compute_dnsmos = false` in training/evaluation flow-ul curent

### 15.7 Agregarea

`evaluate_manifest(...)` din `sebench/training.py:935` agregheaza metricile astfel:

- ignora valorile `NaN`
- raporteaza explicit si count-urile:
  - `pesq_count`
  - `stoi_count`
  - `sisdr_count`
  - `delta_snr_count`
  - `composite_count`
  - `dnsmos_count`

Aceasta este important pentru audit:

- o medie peste putine fisiere valide este interpretabila doar daca vezi si count-ul.

## 16. Artefactele generate de un run

### 16.1 Output root

Fiecare run academic curat trebuie sa aiba propriul `output_root` si propriul `tracking_root`.

Pentru run-ul academic activ acum:

- `output_root = /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs`
- `tracking_root = /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/tracking`

### 16.2 Checkpoint-uri

Teacher:

- `outputs/checkpoints/teacher/<run_name>/model.pt`
- `outputs/checkpoints/teacher/<run_name>/model.final.pt`
- `outputs/checkpoints/teacher/<run_name>/latest_state.pt`
- `outputs/checkpoints/teacher/<run_name>/training_history.csv`
- `outputs/checkpoints/teacher/<run_name>/training_history.json`
- `outputs/checkpoints/teacher/<run_name>/training_history.png`

Stage1 si QAT urmeaza aceeasi conventie in directoarele lor.

### 16.3 Group summaries

Pipeline-ul scrie summary-uri agregate de grup in:

- `outputs/summaries/teacher_training/summary_<timestamp>.json`
- `outputs/summaries/stage1_training/summary_<timestamp>.json`
- `outputs/summaries/qat_training/summary_<timestamp>.json`

Aceste fisiere sunt cele mai importante pentru selectie inter-run.

### 16.4 Teacher cache

Teacher cache:

- `outputs/teacher_cache/teacher_cache.csv`
- `outputs/teacher_cache/summary.json`

### 16.5 Evaluare finala

Evaluarea finala scrie in:

- `outputs/evaluations/<label>/summary.json`
- `outputs/evaluations/<label>/canonical_metrics.csv`
- `outputs/evaluations/<label>/voicebank_val_rank.json`
- `outputs/evaluations/<label>/dns5_val_rank.json`
- `outputs/evaluations/<label>/voicebank_val_select.json`
- `outputs/evaluations/<label>/dns5_val_select.json`
- `outputs/evaluations/<label>/voicebank_test.json`
- `outputs/evaluations/<label>/dns5_test.json`
- `outputs/evaluations/<label>/mcu_rollup.json`

`canonical_metrics.csv` este fisierul cel mai simplu de consumat pentru tabele si plotting extern.

### 16.6 Tracking local

Tracking-ul local are structura:

- `tracking/experiments.json`
- `tracking/runs/<run_id>/meta.json`
- `tracking/runs/<run_id>/params.json`
- `tracking/runs/<run_id>/latest_metrics.json`
- `tracking/runs/<run_id>/metrics_history.jsonl`
- `tracking/runs/<run_id>/artifacts/`

Acest layer este esential pentru audit fin al run-urilor.

## 17. Ce trebuie urmarit in timp real

### 17.1 Fisierele critice

In timpul unui training, fisierele care trebuie urmarite primul sunt:

1. `latest_state.pt`
2. `training_history.csv`
3. `tracking/runs/<run_id>/latest_metrics.json`
4. summary-ul de grup corespunzator dupa terminarea fazei

### 17.2 Ce spune `latest_state.pt`

`latest_state.pt` contine cel putin:

- `epoch`
- `global_step`
- `reason`
- `best_epoch`
- `best_score`
- `epochs_without_improve`
- `history_rows`
- starea optimizer/scheduler/scaler/model

`reason` poate lua valori de tip:

- `periodic`
- `epoch`
- `best`
- `final`
- `failed`
- `interrupted`

Semnificatia practica este directa:

- `periodic`: save intermediar de siguranta
- `epoch`: sfarsit de epoca
- `best`: checkpoint-ul castigator a fost actualizat
- `final`: run terminat
- `failed`: run-ul a crapat, dar starea a fost persistata

### 17.3 Ce spune `training_history.csv`

`training_history.csv` este tabela cea mai utila pentru analiza trendului.

Contine, in functie de faza si eval-urile rulate:

- `epoch`
- `global_step`
- `lr`
- `epoch_seconds`
- `selection_score`
- `improved`
- `epochs_without_improve`
- `train/*`
- `val_rank/*`
- `dns5_val_rank/*`
- `val_select/*`
- `dns5_val_select/*`

### 17.4 Comenzi utile de monitorizare

Pornire run academic complet:

```bash
CUDA_VISIBLE_DEVICES=3 \
/mnt/STORAGE/ulp-stack/venvs/metricgan-gpu/bin/python \
/mnt/STORAGE/ulp-stack/projects/MetricGANplus_Teacher-Student_Speech-Enhancement/repro.py \
  --config /mnt/STORAGE/ulp-stack/projects/MetricGANplus_Teacher-Student_Speech-Enhancement/configs/scenario_combined_datasets_kingston_runtime_academic_clean_20260403_121456.yaml \
  run_all --device cuda:0
```

Monitorizare proces:

```bash
ps -p <PID> -o pid,etimes,%cpu,%mem,cmd
nvidia-smi
```

Status operational rapid:

```bash
/mnt/STORAGE/ulp-stack/venvs/metricgan-gpu/bin/python \
/mnt/STORAGE/ulp-stack/projects/MetricGANplus_Teacher-Student_Speech-Enhancement/scripts/experiment_status.py \
  --run-root /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456 \
  --process-match scenario_combined_datasets_kingston_runtime_academic_clean_20260403_121456.yaml
```

Scriptul raporteaza:

- daca teacher sau student sunt stale
- ultimul `state` din `progress.json`
- `best_val_select_pesq`
- `best_target_gap` fata de pragul cerut
- recomandare operationala (`teacher_running`, `teacher_has_select_signal`, `teacher_below_target_stop_before_student`, etc.)

Monitorizare checkpoint teacher:

```bash
find /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs/checkpoints/teacher -name latest_state.pt -o -name training_history.csv
```

Monitorizare tracking:

```bash
find /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/tracking/runs -maxdepth 2 -type f | sort
```

Monitorizare summary-uri:

```bash
find /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs/summaries -type f | sort
```

`progress.json` in fiecare run:

```bash
find /mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs/checkpoints -name progress.json | sort
```

Acest fisier este heartbeat-ul principal pentru controlul runtime. El contine inclusiv:

- `state`
- `epoch`
- `global_step`
- `best_selection_score`
- `best_epoch`
- `best_target_gap`
- mesajele de progres din `val_rank` si `val_select`

## 18. Controlul reproductibilitatii

Pentru a putea reproduce si audita un experiment, trebuie pastrate impreuna:

- configul exact folosit
- output root-ul izolat
- tracking root-ul izolat
- summary-ul de `prepare_data`
- summary-ul de `teacher_training`
- summary-ul de `stage1_training`
- `canonical_metrics.csv` din evaluarea finala
- checksum sau identificator pentru checkpoint-ul initial de referinta

In mod practic, fara aceste fisiere, comparatia intre run-uri devine ambigua.

## 19. Run-ul academic curent

Run-ul academic curent activ la momentul redactarii acestui document este:

- config: `configs/scenario_combined_datasets_kingston_runtime_academic_clean_20260403_121456.yaml`
- output root: `/mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/outputs`
- tracking root: `/mnt/STORAGE/ulp-stack/results/metricgan/academic_clean_run_20260403_121456/tracking`

Proprietati cheie:

- porneste din `reference/checkpoints/metricgan_plus_native8k_small.pt`
- nu foloseste `resume_checkpoint`
- nu foloseste `resume_training_state`
- nu injecteaza `test` in training
- ruleaza `teacher -> teacher_cache -> stage1 -> evaluate`
- `qat` este oprit automat

Acesta este run-ul de referinta pentru protocolul academic curent.

## 20. Ce este corect stiintific acum si ce nu este inca perfect

### 20.1 Ce este corect

- split-urile sunt disjuncte
- trainingul nu vede `test`
- teacher-ul si studentul sunt selectati pe validation, nu pe test
- `VoiceBank val_select PESQ` este metrica principala de selectie
- `DNS5` este pastrat in training si validation ca guardrail de robustete
- teacher cache-ul este generat din teacher-ul FP32 castigator
- metricele sunt robuste la `NaN` si semnale degenerate

### 20.2 Ce nu este inca perfect

- `VoiceBank test` din runtime este reconstruit din `combined_test_csv` daca manifestul brut lipseste din bundle; asta e acceptabil operational, dar nu ideal pentru o anexare la un benchmark extern
- `DNS5 test` este internal hold-out, nu benchmark oficial challenge
- `phase_c` foloseste un surrogate de PESQ, nu un trainer adversarial MetricGAN+ canonic
- configul curent foloseste un singur seed

Concluzia corecta este:

- protocolul este academic curat pentru cercetare interna serioasa si selectie corecta de model
- pentru claims foarte tari de tip benchmark paper-grade ar trebui, in plus, un test oficial DNS5 si multiple seed-uri

## 21. Ce inovam acum

Aceasta este partea care trebuie pusa in fata cand explicam directia actuala.

### 21.1 Nu mai facem training pe un amestec brut dominat de DNS5

Problema initiala a setului combinat era dominarea optimization-ului de catre DNS5, care este mult mai mare decat VoiceBank.

Inovatia practica este schimbarea protocolului de training fara a arunca date:

- `phase_a`: full mixed union, pentru stabilizare
- `phase_b`: `VoiceBank full + DNS5 replay shards`
- `phase_c`: `VoiceBank-biased + lighter DNS5 replay`

Asta permite simultan:

- folosirea tuturor datelor disponibile
- mentinerea unui target academic comparabil pe VoiceBank
- control explicit al compromisului intre calitate si robustete cross-domain

### 21.2 Selectia este orientata explicit catre tinta reala

Selectorul principal nu este un scor mixt.

Este:

- `VoiceBank val_select PESQ`

Dar DNS5 nu este ignorat; devine guardrail.

Acesta este un punct metodologic bun:

- optimizezi exact ce vrei sa impingi in sus
- fara sa lasi modelul sa colapseze pe celalalt domeniu

### 21.3 Teacher cache-ul pentru student este acum curat

Distilarea foloseste output-urile FP32 ale teacher-ului castigator.

Asta elimina o sursa de degradare artificiala pe care o introducea cache-ul derivat din output-uri cuantizate dinamic.

Impactul practic:

- studentul invata din teacher-ul real, nu dintr-o aproximare mai slaba

### 21.4 Introducem un scurt finetune PESQ-aware, dar controlat

`phase_c` nu este un nou model, nu este un alt teacher family si nu rupe linia existenta.

Este o interventie scurta, controlata, peste teacher-ul curent:

- scurt
- low learning rate
- VoiceBank-biased
- cu surrogate de PESQ invatat offline

Aceasta este inovatia centrala de optimizare a teacher-ului fara a schimba radical paradigma.

### 21.5 Studentul ramane deployment-aware

Desi tinta imediata este cresterea `PESQ`, studentul nu este tratat ca un simplu model mic.

Pipeline-ul pastreaza:

- selectie pe performanta audio
- fallback de familie doar daca e necesar
- artefacte de evaluare curate
- posibilitate de QAT ulterior
- audit de latenta si `mcu_rollup` in evaluarea finala

Asta inseamna ca inovam intr-o directie completa:

- calitate audio mai buna
- protocol academic mai curat
- transfer eficient teacher -> student
- pastrarea directiei de deployment real

## 22. Checklist operational pentru fiecare experiment nou

Inainte de run:

- verifica `resume_checkpoint` si `resume_training_state`
- verifica `output_root` izolat
- verifica `tracking_root` izolat
- verifica `test_csv = None` in configurile de train
- verifica manifestele per-domain si summary-ul `prepare_data`

In timpul teacher-ului:

- urmareste `latest_state.pt`
- urmareste `training_history.csv`
- verifica aparitia `best` checkpoint-urilor
- verifica trendul `VoiceBank val_select PESQ`
- verifica `DNS5 val_select PESQ` fata de guardrail

In timpul studentului:

- verifica existenta `teacher_cache/summary.json`
- verifica `teacher_cache.csv`
- verifica `phase_s1` si `phase_s2`
- verifica daca `causal_s` trece floor-ul sau daca este nevoie de fallback `causal_n6`

La final:

- verifica `outputs/summaries/teacher_training/*.json`
- verifica `outputs/summaries/stage1_training/*.json`
- verifica `outputs/evaluations/<label>/summary.json`
- extrage `canonical_metrics.csv`
- pastreaza configul exact langa rezultate

## 23. Ce trebuie facut in continuare pentru a ridica standardul si mai mult

Daca obiectivul este `publication-grade`, urmatorii pasi tehnici sunt clari:

1. folosirea unui manifest oficial separat pentru `VoiceBank test` in runtime bundle, in loc de reconstructie din `combined_test_csv`
2. introducerea unui `DNS5 test` extern/fix, nu doar internal hold-out
3. rularea a minim `3` seed-uri pe teacher finalist si student finalist
4. comparatie explicita intre:
   - `phase_b` fara PESQ proxy
   - `phase_c` cu PESQ proxy
5. raportarea variatiei inter-seed, nu doar a scorului maxim

Aceasta ar transforma protocolul dintr-un sistem academic corect si bine controlat intr-un benchmark mult mai puternic pentru raportare externa.
