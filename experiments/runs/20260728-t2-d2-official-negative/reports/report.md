# D2-OFFICIAL — negative result

Status: **failed scientific gate; retained for reproducibility**

The fresh, official-parity MetricGAN discriminator selected epoch 1 and
early-stopped at epoch 6. On the untouched 200-record audit it obtained
normalized MAE `0.289481`, Pearson `0.762561` and Spearman `0.776770`.
The 395 eligible local comparisons obtained sign agreement `0.529114` and
delta Spearman `-0.492946`.

The selected checkpoint SHA-256 is
`34112ac7c200245588f0b8832565b883edc8040ae277d032c954690b525b7a66`.
It is evidence only and must not be used to guide a generator.

The package contains the selected model, complete metric history, sanitized
provenance and three diagnostic figures. It excludes VoiceBank+DEMAND audio,
teacher cache, D replay/support and training state.
