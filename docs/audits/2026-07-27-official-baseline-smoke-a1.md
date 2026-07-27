# Official-baseline smoke audit

Run: `20260727-official-baseline-smoke-s0-a1`  
Status: `smoke-passed`, independently audited, verification-only.

## Scope and integrity

- campaign scope: `official_teacher_students_baseline`;
- expected and observed cells: `T0-WB-OFFICIAL`, `S0-WB`, `S0-NB`;
- metric proxies, T1 and S1 cells: absent;
- 3/3 hashed model packages;
- 18/18 reported sample files;
- independent audit issues: 0;
- shared project environment and one NVIDIA GPU.

The teacher is the pinned official SpeechBrain checkpoint. It produced
PESQ-WB 3.3407 and STOI 0.9334 on the four-pair smoke test.

## Cache

The content-addressed Desktop-local cache was validated and reused. It stores
FP16 teacher waveforms and ERB masks for WB and NB. Its manifest binds:

- WB student input/reference/PESQ to 16 kHz/WB/WB;
- NB student input/reference/PESQ to 8 kHz/NB/NB.

The cache contains no noisy or clean input files (`0` files in both input-cache
classes). Dataset paths remain external read-only references.

## Student smoke metrics

| Cell | Protocol | Test PESQ | Test STOI |
|---|---|---:|---:|
| `S0-WB` | WB reference, PESQ-WB | 2.8906 | 0.9206 |
| `S0-NB` | NB reference, PESQ-NB | 3.2869 | 0.9240 |

These four-pair, one-epoch numbers prove wiring only. They are not publication
evidence and are not used to choose a model.

## Decision

The baseline-only controller passes its structural, GPU, cache, bandwidth and
package-audit gates. A clean committed snapshot may launch `run-baseline` on
the frozen full manifests. That full launch must stop after the official
teacher and the two S0 students.

