# KINGSTON Runtime Bundle (MetricGAN + ultra-low-power-se)

Pentru descrierea completa a experimentului academic MetricGAN combinat, incluzand split-uri, antrenare, selectie, monitorizare si criterii de audit, vezi `README_ACADEMIC_EXPERIMENT_CONTROL.md` din radacina proiectului MetricGAN.

Acest folder conține flow-ul complet pentru mutarea pe server cu rulare direct din `/media/$USER/KINGSTON/ulp-stack`, folosind doar date runtime (train/val/test), fără raw inutil.

## Flux recomandat

1. Pe laptop (pregătire bundle pe KINGSTON):
   - `bash runbooks/kingston_runtime/prepare_kingston_runtime_bundle.sh`
2. Pe server (după conectare KINGSTON):
   - `source /media/$USER/KINGSTON/ulp-stack/runbooks/env_server.sh`
   - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/check_env_gpu.sh`
   - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/install_requirements_server.sh`
   - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/build_ultra_combined_cache.sh`
   - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/run_smoke_all.sh`
   - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/run_full_dual_gpu.sh`

## Ce face bundle-ul

- Copiază proiectele:
  - `MetricGANplus_Teacher-Student_Speech-Enhancement`
  - `ultra-low-power-se`
- Copiază strict audio folosit în manifestele staged (`train/val_rank/val_select/test`).
- Rescrie manifestele către rădăcina KINGSTON.
- Copiază checkpoint-ul teacher pentru resume (`latest_state.pt` + `model.pt`).
- Generează config runtime MetricGAN pe KINGSTON.
- Creează structură separată de rezultate per run.

## Note importante

- Scripturile de install folosesc exclusiv interpreterul din venv (`VENV_PYTHON`) și nu instalează global.
- `run_full_dual_gpu.sh` pornește:
  - pipeline-ul complet MetricGAN (`prepare_data -> teacher -> teacher_cache -> stage1 -> evaluate`) pe GPU0
  - ultra-low-power-se pe GPU1
- Pentru export rezultate articol:
  - `bash /media/$USER/KINGSTON/ulp-stack/runbooks/export_results_bundle.sh`
