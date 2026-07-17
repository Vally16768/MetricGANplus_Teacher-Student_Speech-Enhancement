from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sebench.stm32_models import waveform_to_erb_mask


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
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
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
        }


def save_pesq_proxy_checkpoint(path: str | Path, model: PESQProxyRegressor) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config_dict(), "state_dict": model.state_dict()}, out_path)


def load_pesq_proxy_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PESQProxyRegressor:
    payload = torch.load(Path(path), map_location="cpu")
    config = dict(payload.get("config") or {})
    model = PESQProxyRegressor(**config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    ):
        super().__init__()
        self.recipe = recipe.upper()
        if self.recipe not in {"D1", "D2", "T0", "T0_PESQ"}:
            raise ValueError(
                f"Unsupported loss recipe for the standalone project: {recipe}. "
                "Supported recipes: T0, T0_PESQ, D1, D2."
            )
        self.erb_bands = erb_bands
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.wave_loss = nn.SmoothL1Loss(beta=0.5)
        self.teacher_mask_loss = nn.L1Loss()
        self.complex_loss = ComplexSTFTLoss()
        self.sisdr_loss = SISDRLoss()
        self.pesq_proxy = pesq_proxy

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
            spectral = self.complex_loss(enhanced, clean)
            sisdr = self.sisdr_loss(enhanced, clean)
            t0_total = 0.70 * spectral + 0.25 * wave + 0.05 * sisdr
            pesq_proxy_loss = zero
            predicted_pesq = zero
            total = t0_total
            if self.recipe == "T0_PESQ":
                if self.pesq_proxy is None:
                    raise ValueError("Loss recipe T0_PESQ requires a frozen PESQ proxy model.")
                predicted_pesq = self.pesq_proxy(noisy, enhanced, clean).mean()
                pesq_proxy_loss = -predicted_pesq
                total = 0.60 * t0_total + 0.25 * pesq_proxy_loss + 0.15 * sisdr
            return LossBreakdown(
                total=total,
                wave=wave,
                spectral=spectral,
                sisdr=sisdr,
                noise_gate=zero,
                speech_preserve=zero,
                teacher_mask=zero,
                teacher_wave=zero,
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
        if self.recipe == "D2":
            sisdr = self.sisdr_loss(enhanced, clean)
            total = total + 0.05 * sisdr

        return LossBreakdown(
            total=total,
            wave=wave,
            spectral=spectral,
            sisdr=sisdr,
            noise_gate=zero,
            speech_preserve=zero,
            teacher_mask=teacher_mask,
            teacher_wave=teacher_wave,
            pesq_proxy=zero,
            predicted_pesq=zero,
        )
