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

## Rulări promovate

| Run | Scope | Status |
|---|---|---|
| [`20260727-converged-s0-baseline-v2`](runs/20260727-converged-s0-baseline-v2/) | T0 oficial + S0-WB/S0-NB convergenți; evaluare la lungimea reală | canonic, auditat |
| [`20260727-converged-s0-baseline-v1`](runs/20260727-converged-s0-baseline-v1/) | aceleași modele, evaluare istorică sensibilă la padding | supersedat; nu se citează |

`import_manifest.json` este inventarul complet al fișierelor publicate și le
leagă de dimensiune și SHA-256. Orice modificare ulterioară este detectată de
`campaign.py audit-run`.
