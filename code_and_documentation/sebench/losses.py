from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from sebench.bandwidth import resolve_bandwidth, validate_frontend
from sebench.erb import waveform_to_erb_mask


class SISDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)

        clean = clean - clean.mean(dim=-1, keepdim=True)
        enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)

        proj = (enhanced * clean).sum(dim=-1, keepdim=True) * clean
        proj = proj / (clean.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        noise = enhanced - proj

        ratio = (proj.pow(2).sum(dim=-1) + self.eps) / (noise.pow(2).sum(dim=-1) + self.eps)
        return -10.0 * torch.log10(ratio).mean()


class ComplexSTFTLoss(nn.Module):
    def __init__(self, n_ffts: tuple[int, ...] = (256, 512, 1024)):
        super().__init__()
        self.n_ffts = n_ffts

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)

        total = enhanced.new_tensor(0.0)
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=enhanced.device)
            enh_spec = torch.stft(
                enhanced,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            clean_spec = torch.stft(
                clean,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            real_l1 = torch.mean(torch.abs(enh_spec.real - clean_spec.real))
            imag_l1 = torch.mean(torch.abs(enh_spec.imag - clean_spec.imag))
            mag_l1 = torch.mean(torch.abs(enh_spec.abs() - clean_spec.abs()))
            total = total + (real_l1 + imag_l1 + mag_l1) / 3.0
        return total / float(len(self.n_ffts))


class PESQProxyRegressor(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        hidden_channels: int = 32,
        projection_dim: int = 64,
        bandwidth: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.bandwidth = validate_frontend(
            bandwidth,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        ).name
        self.hidden_channels = int(hidden_channels)
        self.projection_dim = int(projection_dim)
        self.register_buffer("_window", torch.hann_window(self.win_length), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 2 * 4 * 4, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1),
        )

    def _logmag(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        window = self._window.to(device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return torch.log1p(spec.abs().clamp_min(1e-6))

    def forward(self, noisy: torch.Tensor, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        clean_mag = self._logmag(clean)
        enh_mag = self._logmag(enhanced)
        noisy_mag = self._logmag(noisy)
        features = torch.stack(
            [
                clean_mag,
                enh_mag,
                noisy_mag,
                torch.abs(clean_mag - enh_mag),
                torch.abs(clean_mag - noisy_mag),
                torch.abs(enh_mag - noisy_mag),
            ],
            dim=1,
        )
        raw = self.head(self.encoder(features)).squeeze(-1)
        return 4.5 * torch.sigmoid(raw)

    def config_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "hidden_channels": self.hidden_channels,
            "projection_dim": self.projection_dim,
            "bandwidth": self.bandwidth,
        }


def _metricgan_layer(
    in_size: int,
    out_size: int | None = None,
    *,
    layer_type: type[nn.Module] = nn.Linear,
    **kwargs: Any,
) -> nn.Module:
    """SpeechBrain MetricGAN layer initialization with spectral normalization."""
    if out_size is None:
        out_size = in_size
    layer = spectral_norm(layer_type(in_size, out_size, **kwargs))
    nn.init.xavier_uniform_(layer.weight_orig, gain=1.0)
    nn.init.zeros_(layer.bias)
    return layer


class SpeechBrainMetricDiscriminator(nn.Module):
    """MetricGAN discriminator matching SpeechBrain's published architecture.

    The native network estimates normalized PESQ. ``forward`` exposes raw PESQ
    for compatibility with :class:`MetricGANGeneratorObjective`; discriminator
    training uses :meth:`normalized_score` directly.
    """

    checkpoint_kind = "speechbrain_metric_discriminator"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        bandwidth: str = "wb",
        kernel_size: tuple[int, int] = (5, 5),
        base_channels: int = 15,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.bandwidth = resolve_bandwidth(
            bandwidth,
            sample_rate=self.sample_rate,
        ).name
        if (self.n_fft, self.hop_length, self.win_length) != (512, 256, 512):
            raise ValueError(
                "SpeechBrain MetricGAN discriminator requires frontend "
                "512/256/512."
            )
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.base_channels = int(base_channels)
        self.register_buffer(
            "_window",
            torch.hamming_window(self.win_length),
            persistent=False,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=2, momentum=0.01)
        self.conv1 = _metricgan_layer(
            2,
            self.base_channels,
            layer_type=nn.Conv2d,
            kernel_size=self.kernel_size,
        )
        self.conv2 = _metricgan_layer(
            self.base_channels,
            layer_type=nn.Conv2d,
            kernel_size=self.kernel_size,
        )
        self.conv3 = _metricgan_layer(
            self.base_channels,
            layer_type=nn.Conv2d,
            kernel_size=self.kernel_size,
        )
        self.conv4 = _metricgan_layer(
            self.base_channels,
            layer_type=nn.Conv2d,
            kernel_size=self.kernel_size,
        )
        self.linear1 = _metricgan_layer(self.base_channels, 50)
        self.linear2 = _metricgan_layer(50, 10)
        self.linear3 = _metricgan_layer(10, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.3)

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        window = self._window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        # SpeechBrain's MetricGAN recipe applies
        # ``log1p(spectral_magnitude(stft, power=0.5))``. Its helper first
        # forms real**2 + imag**2 and then raises that power spectrum to 0.5,
        # which is the ordinary complex magnitude (not sqrt(magnitude)).
        # SpeechBrain also exposes the frontend as [batch, time, frequency].
        return torch.log1p(spectrum.abs()).transpose(1, 2)

    def normalized_score(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            if candidate.shape[0] != reference.shape[0]:
                raise ValueError("Candidate/reference batch sizes must match.")
            if int(lengths.numel()) != int(candidate.shape[0]):
                raise ValueError("One absolute sample length is required per item.")
            scores = []
            for index, length_value in enumerate(lengths.detach().cpu().tolist()):
                sample_count = int(length_value)
                if sample_count <= 0 or sample_count > int(candidate.shape[-1]):
                    raise ValueError("Metric discriminator length is out of bounds.")
                scores.append(
                    self.normalized_score(
                        candidate[index : index + 1, ..., :sample_count],
                        reference[index : index + 1, ..., :sample_count],
                    )
                )
            return torch.cat(scores, dim=0)
        features = torch.stack(
            [self._features(candidate), self._features(reference)],
            dim=1,
        )
        value = self.batch_norm(features)
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            value = self.activation(layer(value))
        value = torch.mean(value, dim=(2, 3))
        value = self.activation(self.linear1(value))
        value = self.activation(self.linear2(value))
        return self.linear3(value).squeeze(-1)

    def forward(
        self,
        noisy: torch.Tensor,
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del noisy
        return 5.0 * self.normalized_score(candidate, reference) - 0.5

    def config_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "bandwidth": self.bandwidth,
            "kernel_size": self.kernel_size,
            "base_channels": self.base_channels,
        }


def save_pesq_proxy_checkpoint(
    path: str | Path,
    model: PESQProxyRegressor | SpeechBrainMetricDiscriminator,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": getattr(model, "checkpoint_kind", "pesq_proxy_regressor"),
            "config": model.config_dict(),
            "state_dict": model.state_dict(),
        },
        out_path,
    )


def load_pesq_proxy_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    freeze: bool = True,
) -> PESQProxyRegressor | SpeechBrainMetricDiscriminator:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = dict(payload.get("config") or {})
    kind = str(payload.get("kind") or "pesq_proxy_regressor")
    if kind == SpeechBrainMetricDiscriminator.checkpoint_kind:
        model: PESQProxyRegressor | SpeechBrainMetricDiscriminator = (
            SpeechBrainMetricDiscriminator(**config)
        )
    elif kind == "pesq_proxy_regressor":
        model = PESQProxyRegressor(**config)
    else:
        raise ValueError(f"Unsupported PESQ proxy checkpoint kind: {kind}")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)
    return model


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    wave: torch.Tensor
    spectral: torch.Tensor
    sisdr: torch.Tensor
    noise_gate: torch.Tensor
    speech_preserve: torch.Tensor
    teacher_mask: torch.Tensor
    teacher_wave: torch.Tensor
    pesq_proxy: torch.Tensor
    predicted_pesq: torch.Tensor


class MetricGANGeneratorObjective(nn.Module):
    """Bounded MetricGAN-style objective backed by a frozen metric proxy.

    For enhancement, ``source`` is the noisy waveform. For a future TTS
    experiment it may be omitted, but the proxy must first be recalibrated on
    outputs from that synthesis domain.

    The official MetricGAN recipe normalizes PESQ from ``[-0.5, 4.5]`` to
    ``[0, 1]`` and minimizes MSE to the clean-speech target score ``1``.
    Maximizing an unbounded predicted PESQ instead lets a generator exploit a
    frozen proxy outside its calibration distribution.
    """

    def __init__(
        self,
        metric_proxy: nn.Module,
        *,
        metric_min: float = -0.5,
        metric_max: float = 4.5,
    ) -> None:
        super().__init__()
        if metric_max <= metric_min:
            raise ValueError("metric_max must be greater than metric_min.")
        self.metric_proxy = metric_proxy
        self.metric_min = float(metric_min)
        self.metric_max = float(metric_max)
        self.metric_proxy.eval()
        for parameter in self.metric_proxy.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proxy_source = torch.zeros_like(candidate) if source is None else source
        predicted_quality = self.metric_proxy(proxy_source, candidate, reference)
        normalized_quality = (predicted_quality - self.metric_min) / (
            self.metric_max - self.metric_min
        )
        target_quality = torch.ones_like(normalized_quality)
        return torch.mean((normalized_quality - target_quality) ** 2), predicted_quality


class MetricGANFeatureLoss(nn.Module):
    """Official MetricGAN log-spectral feature MSE."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer(
            "_window",
            torch.hamming_window(self.win_length),
            persistent=False,
        )

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        window = self._window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return torch.log1p(spectrum.abs().clamp_min(1e-8).sqrt())

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return torch.mean((self._features(enhanced) - self._features(clean)) ** 2)


class CompositeEnhancementLoss(nn.Module):
    def __init__(
        self,
        recipe: str,
        sample_rate: int = 16000,
        erb_bands: int = 32,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        pesq_proxy: nn.Module | None = None,
        metric_proxy_weight: float = 0.25,
        teacher_anchor_weight: float = 0.75,
    ):
        super().__init__()
        self.recipe = recipe.upper()
        if self.recipe not in {
            "D1",
            "D1_PESQ",
            "D2",
            "D2_PESQ",
            "T0",
            "T0_PESQ",
        }:
            raise ValueError(
                f"Unsupported loss recipe for the standalone project: {recipe}. "
                "Supported recipes: T0, T0_PESQ, D1, D1_PESQ, D2, D2_PESQ."
            )
        self.erb_bands = erb_bands
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.wave_loss = nn.SmoothL1Loss(beta=0.5)
        self.teacher_mask_loss = nn.L1Loss()
        self.complex_loss = ComplexSTFTLoss()
        self.metricgan_feature_loss = MetricGANFeatureLoss(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.sisdr_loss = SISDRLoss()
        self.pesq_proxy = pesq_proxy
        self.metric_proxy_weight = float(metric_proxy_weight)
        if not 0.0 <= self.metric_proxy_weight <= 1.0:
            raise ValueError("metric_proxy_weight must be in [0, 1].")
        self.teacher_anchor_weight = float(teacher_anchor_weight)
        if not 0.0 <= self.teacher_anchor_weight <= 1.0:
            raise ValueError("teacher_anchor_weight must be in [0, 1].")
        self.metric_objective = (
            MetricGANGeneratorObjective(pesq_proxy)
            if pesq_proxy is not None
            else None
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        epoch: int,
        total_epochs: int,
        teacher_wav: torch.Tensor | None = None,
        teacher_mask_erb: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del epoch, total_epochs

        wave = self.wave_loss(enhanced, clean)
        zero = enhanced.new_tensor(0.0)
        if self.recipe in {"T0", "T0_PESQ"}:
            spectral = self.metricgan_feature_loss(enhanced, clean)
            sisdr = self.sisdr_loss(enhanced, clean)
            teacher_wave = zero
            t0_total = spectral
            if teacher_wav is not None:
                teacher_wave = self.wave_loss(enhanced, teacher_wav)
                t0_total = (
                    (1.0 - self.teacher_anchor_weight) * spectral
                    + self.teacher_anchor_weight * teacher_wave
                )
            pesq_proxy_loss = zero
            predicted_pesq = zero
            total = t0_total
            if self.recipe == "T0_PESQ":
                if self.pesq_proxy is None:
                    raise ValueError("Loss recipe T0_PESQ requires a frozen PESQ proxy model.")
                pesq_proxy_loss, predictions = self.metric_objective(
                    enhanced,
                    clean,
                    source=noisy,
                )
                predicted_pesq = predictions.mean()
                total = (
                    (1.0 - self.metric_proxy_weight) * t0_total
                    + self.metric_proxy_weight * pesq_proxy_loss
                )
            return LossBreakdown(
                total=total,
                wave=wave,
                spectral=spectral,
                sisdr=sisdr,
                noise_gate=zero,
                speech_preserve=zero,
                teacher_mask=zero,
                teacher_wave=teacher_wave,
                pesq_proxy=pesq_proxy_loss,
                predicted_pesq=predicted_pesq,
            )

        if teacher_wav is None or teacher_mask_erb is None:
            raise ValueError(f"Loss recipe {self.recipe} requires teacher_wav and teacher_mask_erb.")

        student_mask = waveform_to_erb_mask(
            noisy,
            enhanced,
            erb_bands=self.erb_bands,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        teacher_mask = self.teacher_mask_loss(student_mask, teacher_mask_erb)
        teacher_wave = self.complex_loss(enhanced, teacher_wav)
        spectral = self.complex_loss(enhanced, clean)
        sisdr = enhanced.new_tensor(0.0)

        total = 0.60 * teacher_mask + 0.25 * teacher_wave + 0.15 * spectral
        base_recipe = self.recipe.removesuffix("_PESQ")
        if base_recipe == "D2":
            sisdr = self.sisdr_loss(enhanced, clean)
            total = total + 0.05 * sisdr

        pesq_proxy_loss = zero
        predicted_pesq = zero
        if self.recipe.endswith("_PESQ"):
            if self.metric_objective is None:
                raise ValueError(
                    f"Loss recipe {self.recipe} requires a frozen PESQ proxy model."
                )
            pesq_proxy_loss, predictions = self.metric_objective(
                enhanced,
                clean,
                source=noisy,
            )
            predicted_pesq = predictions.mean()
            total = total + self.metric_proxy_weight * pesq_proxy_loss

        return LossBreakdown(
            total=total,
            wave=wave,
            spectral=spectral,
            sisdr=sisdr,
            noise_gate=zero,
            speech_preserve=zero,
            teacher_mask=teacher_mask,
            teacher_wave=teacher_wave,
            pesq_proxy=pesq_proxy_loss,
            predicted_pesq=predicted_pesq,
        )
