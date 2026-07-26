"""ERB frontend utilities shared by MetricGAN+ losses and teacher caches."""

from __future__ import annotations

import torch

from sebench.postfilters import resolve_postfilter_config, spectral_gate_waveform


WB_SAMPLE_RATE = 16_000
WB_N_FFT = 512
WB_HOP_LENGTH = 160
WB_WIN_LENGTH = 320
DEFAULT_ERB_BANDS = 32


def frontend_defaults_for_sample_rate(sample_rate: int) -> tuple[int, int, int]:
    if int(sample_rate) <= 8_000:
        return 256, 80, 160
    return WB_N_FFT, WB_HOP_LENGTH, WB_WIN_LENGTH


def padded_frame_count(
    length: int,
    *,
    n_fft: int = WB_N_FFT,
    hop_length: int = WB_HOP_LENGTH,
) -> int:
    del n_fft
    return max(1, 1 + max(0, length // hop_length))


def _erb_scale(freq_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * freq_hz.clamp_min(0.0))


def _inv_erb_scale(erb_value: torch.Tensor) -> torch.Tensor:
    return (10.0 ** (erb_value / 21.4) - 1.0) / 0.00437


def build_erb_filterbank(
    *,
    n_fft: int = WB_N_FFT,
    sample_rate: int = WB_SAMPLE_RATE,
    bands: int = DEFAULT_ERB_BANDS,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    freq_bins = n_fft // 2 + 1
    freqs = torch.linspace(
        0.0,
        sample_rate / 2.0,
        freq_bins,
        device=device,
        dtype=dtype or torch.float32,
    )
    edges = torch.linspace(
        float(_erb_scale(freqs[:1])),
        float(_erb_scale(freqs[-1:])),
        bands + 2,
        device=device,
        dtype=freqs.dtype,
    )
    hz_edges = _inv_erb_scale(edges)
    bank = torch.zeros(bands, freq_bins, device=device, dtype=freqs.dtype)
    for band_idx in range(bands):
        left, center, right = hz_edges[band_idx : band_idx + 3]
        rising = torch.clamp(
            (freqs - left) / (center - left + 1e-6),
            min=0.0,
            max=1.0,
        )
        falling = torch.clamp(
            (right - freqs) / (right - center + 1e-6),
            min=0.0,
            max=1.0,
        )
        bank[band_idx] = torch.minimum(rising, falling)
    return bank / bank.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def waveform_to_stft(
    wav: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    if wav.ndim == 3:
        wav = wav.squeeze(1)
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def project_mag_to_erb(
    magnitude: torch.Tensor,
    erb_filterbank: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bft,ef->bet", magnitude, erb_filterbank)


def waveform_to_erb_mask(
    noisy: torch.Tensor,
    enhanced: torch.Tensor,
    *,
    erb_bands: int = DEFAULT_ERB_BANDS,
    sample_rate: int = WB_SAMPLE_RATE,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    default_fft, default_hop, default_win = frontend_defaults_for_sample_rate(
        sample_rate
    )
    n_fft = default_fft if n_fft is None else n_fft
    hop_length = default_hop if hop_length is None else hop_length
    win_length = default_win if win_length is None else win_length
    noisy_spec = waveform_to_stft(
        noisy,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    enhanced_spec = waveform_to_stft(
        enhanced,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    bank = build_erb_filterbank(
        n_fft=n_fft,
        sample_rate=sample_rate,
        bands=erb_bands,
        device=noisy_spec.device,
        dtype=noisy_spec.real.dtype,
    )
    noisy_erb = project_mag_to_erb(noisy_spec.abs().clamp_min(1e-5), bank)
    enhanced_erb = project_mag_to_erb(enhanced_spec.abs().clamp_min(1e-5), bank)
    return (enhanced_erb / noisy_erb.clamp_min(1e-5)).clamp(0.0, 2.0)


def compute_spectral_gating_guidance(
    noisy: torch.Tensor,
    *,
    erb_bands: int = DEFAULT_ERB_BANDS,
    sample_rate: int = WB_SAMPLE_RATE,
    preset: str = "medium",
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    config = resolve_postfilter_config("sg_input_floor", preset)
    gated = spectral_gate_waveform(noisy, noisy, config)
    return waveform_to_erb_mask(
        noisy,
        gated,
        erb_bands=erb_bands,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
