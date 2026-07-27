# Bounded-teacher smoke audit

Run: `20260727-bounded-teacher-smoke-s0-a1`  
Status: `smoke-passed`, independently audited, verification-only.

The run used clean commit `27838d9`, frozen VoiceBank+DEMAND smoke manifests
and the shared CUDA environment. It completed 7/7 cells, produced 7/7 hashed
model packages and retained 42/42 reported sample files. The independent audit
found zero issues.

The safety correction removed the prior teacher collapse:

| T1 branch | Epoch-0 PESQ-WB | Epoch-1 PESQ-WB | Delta |
|---|---:|---:|---:|
| control | 3.0460 | 3.0456 | -0.0004 |
| bounded metric | 3.0460 | 3.0453 | -0.0007 |

This smoke has only four validation pairs. It proves bounded behavior and
correct T0-cache anchoring, not quality improvement. Neither candidate met the
required +0.01 PESQ-WB gate, so best-checkpoint selection restored T0 and S0/S1
reused the same content-addressed local cache. Student deltas were below
0.00002 PESQ.

The smoke proxy remained intentionally underpowered (12/6 candidates, negative
held-out correlation). The next valid test is a clean pilot with the larger
384/96 proxy support. Full training remains blocked.
