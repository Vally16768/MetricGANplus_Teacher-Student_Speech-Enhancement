# Full campaign `20260727-full-wbnb-s0-a1` — controlled stop

## Disposition

Status: `stopped-by-user`, `invalid-for-promotion`, artifacts preserved.

The campaign started from clean commit
`4fee1e386e8433311cd0754900d30f117fe26325`. It was interrupted with SIGINT
during epoch 14 validation of `S-WB-BASE`. No training process remains and no
dataset file was modified. The ignored local run directory is retained in full;
its size at the stop point was 9.4 GiB.

## Completed evidence

- all four VoiceBank+DEMAND manifests passed the split audit with zero pair or
  clean-identity overlap;
- `T-WB-ANCHOR`, `T-WB-BASE`, `T-WB-METRIC`, teacher selection, the dual WB/NB
  teacher cache and both metric proxies completed;
- `T-WB-BASE` was correctly selected on `val_select`: PESQ-WB `2.214801`;
- `T-WB-METRIC` showed proxy exploitation and was rejected:
  `val_select` PESQ-WB `2.016112`, test PESQ-WB `2.375448`, versus baseline
  test PESQ-WB `2.528984`;
- WB proxy held-out calibration: MAE `0.198475`, Pearson `0.979854`, Spearman
  `0.954408`;
- NB proxy held-out calibration: MAE `0.185903`, Pearson `0.975788`, Spearman
  `0.968028`.

## Why the WB student was stopped

The one-layer 96-unit WB student did learn numerically, but its quality was
insufficient for the intended branch. Training loss fell from `0.123468` at
epoch 1 to `0.064293` at epoch 13, and `val_select` PESQ-WB rose from
`1.837460` to a best `1.996045` at epoch 12. It then fell to `1.979789` at
epoch 13. The best student remained `0.218756` PESQ below the selected WB
teacher on the same `val_select` support.

This is evidence of learning with inadequate capacity/result, not evidence that
the optimization graph was disconnected.

## Architecture recovery decision

The student used by the earlier MP-SENet teacher–student campaign was traced to
the local source family `metricgan_plus_native8k_causal_max`, not to the full
parallel magnitude/phase MP-SENet generator. Its exact student settings were:

- unidirectional GRU, hidden size `160`, three layers;
- linear projection `224`;
- Hamming-window STFT;
- eight-frame sequence deployment accounting;
- fixed 16 ms STFT lookahead;
- WB: 16 kHz, 512 FFT bins, `604386` parameters;
- NB: 8 kHz, 256 FFT bins, `514018` parameters.

The recovered local model source SHA-256 is
`d6b457fdab7fa1dde2afe94688259dbcebdd40dd78f0935a1d1b70e5376b33e8`.
The WB and NB source-config hashes are respectively
`483db8ddaabac87569d24196239097e02d98055be0d13ef44328291025b16560`
and
`41c6e543994ffd6caaf964980a4023f0f03fad2be12562316fec3112b1e39b55`.

Only the architecture is transferred. Historical weights are not reused
because their training provenance includes data outside the current
VoiceBank+DEMAND-only contract.

The official MP-SENet source snapshot observed locally is commit
`89932cfe90d1dacb8e170e4a331d762462c21792`. It confirms that the full teacher
architecture is a separate magnitude/phase codec with time-frequency modeling;
it is not being copied into this MetricGAN+ student.

## Recoverability

- best partial `S-WB-BASE` package SHA-256:
  `2c0af0a61d9f9672da8eed929d618d83727cb4c732e8e7e941aa9a294ebad507`;
- interrupted training-state SHA-256:
  `744e2fe65fe8f245b6951f01f500549c62b64839105dd2e71ac6da27a6dec8c3`;
- campaign status remains `failed` at the machine layer because SIGINT is an
  exception; this audit supplies the human disposition `stopped-by-user`;
- none of the partial artifacts may be promoted or used as final article
  evidence.
