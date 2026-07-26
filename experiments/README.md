# Experimente

`runs/` conține numai exporturi imuabile și verificate ale rulărilor
MetricGAN+ teacher–student.

Structura recomandată:

```text
runs/<run_id>/
  provenance/
    config_resolved.yaml
    command.txt
    provenance.json
  logs/
  metrics/
  reports/
  models/
  import_manifest.json
```

Rulările în desfășurare stau în `local/runs/` (ignorat de Git). După finalizare
se exportă într-un `run_id` nou. Datasetul, audio și `teacher_cache` nu se copiază
aici.
