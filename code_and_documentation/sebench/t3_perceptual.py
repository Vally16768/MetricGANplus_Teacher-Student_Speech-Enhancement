"""Direct perceptual objectives for the predeclared T3 teacher study.

The module is intentionally separate from the legacy student losses.  It
implements the matched E1 supervised control and E2 PESQ-inspired ablation
without changing the frozen T0 teacher or the VoiceBank+DEMAND inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

try:
    from torch_pesq import PesqLoss
except ImportError as exc:  # pragma: no cover - exercised by environment guard
    raise ImportError(
        "T3 requires the pinned `torch-pesq==0.1.2` package in shared-venv."
    ) from exc


T3_SAMPLE_RATE = 16_000
T3_RESOLUTIONS = ((256, 64, 256), (512, 128, 512), (1024, 256, 1024))
T3_OFFICIAL_FRONTEND = (512, 256, 512)


def _as_waveform_batch(waveform: torch.Tensor, *, name: str) -> torch.Tensor:
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    if waveform.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time] or [batch, 1, time].")
    if not waveform.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    return waveform


def _sample_lengths(
    lengths: torch.Tensor | Iterable[int] | None,
    *,
    batch_size: int,
    padded_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.full(
            (batch_size,),
            padded_samples,
            dtype=torch.long,
            device=device,
        )
    values = torch.as_tensor(lengths, dtype=torch.long, device=device)
    if values.ndim != 1 or values.numel() != batch_size:
        raise ValueError("lengths must contain one integer sample count per utterance.")
    if bool(torch.any(values < 1)) or bool(torch.any(values > padded_samples)):
        raise ValueError("Every true length must be in [1, padded time dimension].")
    return values


def _validate_aligned(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate = _as_waveform_batch(candidate, name=candidate_name)
    reference = _as_waveform_batch(reference, name=reference_name)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"{candidate_name} and {reference_name} must have identical shapes; "
            f"got {tuple(candidate.shape)} and {tuple(reference.shape)}."
        )
    return candidate, reference


def _stft_magnitude(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.to(device=waveform.device, dtype=waveform.dtype),
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).abs()


class MultiResolutionSTFTLoss(nn.Module):
    """True-length multi-resolution magnitude and log-magnitude L1 loss."""

    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = T3_RESOLUTIONS,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not resolutions:
            raise ValueError("At least one STFT resolution is required.")
        self.resolutions = tuple(tuple(int(value) for value in item) for item in resolutions)
        self.eps = float(eps)
        for index, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
            if min(n_fft, hop_length, win_length) < 1 or win_length > n_fft:
                raise ValueError(f"Invalid STFT resolution: {self.resolutions[index]}.")
            self.register_buffer(
                f"_window_{index}",
                torch.hann_window(win_length),
                persistent=False,
            )

    def forward(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        *,
        lengths: torch.Tensor | Iterable[int] | None = None,
    ) -> torch.Tensor:
        candidate, reference = _validate_aligned(candidate, reference)
        true_lengths = _sample_lengths(
            lengths,
            batch_size=candidate.shape[0],
            padded_samples=candidate.shape[-1],
            device=candidate.device,
        )
        utterance_losses: list[torch.Tensor] = []
        for row, length_tensor in enumerate(true_lengths):
            length = int(length_tensor.item())
            estimate = candidate[row : row + 1, :length]
            target = reference[row : row + 1, :length]
            resolution_losses: list[torch.Tensor] = []
            for index, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
                window = getattr(self, f"_window_{index}")
                estimate_mag = _stft_magnitude(
                    estimate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                )
                target_mag = _stft_magnitude(
                    target,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                )
                magnitude = torch.mean(torch.abs(estimate_mag - target_mag))
                magnitude = magnitude / (torch.mean(target_mag) + self.eps)
                log_magnitude = torch.mean(
                    torch.abs(
                        torch.log1p(estimate_mag) - torch.log1p(target_mag)
                    )
                )
                resolution_losses.append(0.5 * (magnitude + log_magnitude))
            utterance_losses.append(torch.stack(resolution_losses).mean())
        return torch.stack(utterance_losses).mean()


class TrueLengthSISDRLoss(nn.Module):
    """Negative SI-SDR evaluated per true utterance length."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        *,
        lengths: torch.Tensor | Iterable[int] | None = None,
    ) -> torch.Tensor:
        candidate, reference = _validate_aligned(candidate, reference)
        true_lengths = _sample_lengths(
            lengths,
            batch_size=candidate.shape[0],
            padded_samples=candidate.shape[-1],
            device=candidate.device,
        )
        losses: list[torch.Tensor] = []
        for row, length_tensor in enumerate(true_lengths):
            length = int(length_tensor.item())
            estimate = candidate[row, :length]
            target = reference[row, :length]
            estimate = estimate - estimate.mean()
            target = target - target.mean()
            target_energy = torch.sum(target.square())
            projection = torch.sum(estimate * target) * target
            projection = projection / (target_energy + self.eps)
            residual = estimate - projection
            ratio = (torch.sum(projection.square()) + self.eps) / (
                torch.sum(residual.square()) + self.eps
            )
            losses.append(-10.0 * torch.log10(ratio))
        return torch.stack(losses).mean()


class T0LogMagnitudeAnchorLoss(nn.Module):
    """Trust-region loss against T0 using the pinned MetricGAN+ WB frontend."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = T3_OFFICIAL_FRONTEND
        self.eps = float(eps)
        self.register_buffer(
            "_window",
            torch.hamming_window(self.win_length),
            persistent=False,
        )

    def forward(
        self,
        candidate: torch.Tensor,
        teacher_t0: torch.Tensor,
        *,
        lengths: torch.Tensor | Iterable[int] | None = None,
    ) -> torch.Tensor:
        candidate, teacher_t0 = _validate_aligned(
            candidate,
            teacher_t0,
            reference_name="teacher_t0",
        )
        true_lengths = _sample_lengths(
            lengths,
            batch_size=candidate.shape[0],
            padded_samples=candidate.shape[-1],
            device=candidate.device,
        )
        losses: list[torch.Tensor] = []
        for row, length_tensor in enumerate(true_lengths):
            length = int(length_tensor.item())
            estimate_mag = _stft_magnitude(
                candidate[row : row + 1, :length],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self._window,
            )
            teacher_mag = _stft_magnitude(
                teacher_t0[row : row + 1, :length],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self._window,
            )
            losses.append(
                torch.mean(
                    torch.abs(
                        torch.log1p(estimate_mag.clamp_min(self.eps))
                        - torch.log1p(teacher_mag.clamp_min(self.eps))
                    )
                )
            )
        return torch.stack(losses).mean()


class DifferentiablePESQInspiredLoss(nn.Module):
    """Reviewed torch-pesq surrogate with an explicit WB/16 kHz contract."""

    def __init__(
        self,
        *,
        sample_rate: int = T3_SAMPLE_RATE,
        silence_rms: float = 1e-5,
        minimum_samples: int = 5_376,
    ) -> None:
        super().__init__()
        if int(sample_rate) != T3_SAMPLE_RATE:
            raise ValueError("T3 teacher PMSQE is WB-only and requires 16 kHz audio.")
        self.sample_rate = int(sample_rate)
        self.silence_rms = float(silence_rms)
        self.minimum_samples = int(minimum_samples)
        if self.minimum_samples < 5_376:
            raise ValueError("minimum_samples must provide at least 20 PESQ frames.")
        self.loss = PesqLoss(
            factor=1.0,
            sample_rate=self.sample_rate,
            nbarks=49,
            n_fft=512,
            win_length=512,
            hop_length=256,
        )

    def forward(
        self,
        candidate: torch.Tensor,
        reference: torch.Tensor,
        *,
        lengths: torch.Tensor | Iterable[int] | None = None,
    ) -> torch.Tensor:
        candidate, reference = _validate_aligned(candidate, reference)
        true_lengths = _sample_lengths(
            lengths,
            batch_size=candidate.shape[0],
            padded_samples=candidate.shape[-1],
            device=candidate.device,
        )
        losses: list[torch.Tensor] = []
        for row, length_tensor in enumerate(true_lengths):
            length = int(length_tensor.item())
            estimate = candidate[row : row + 1, :length].float()
            target = reference[row : row + 1, :length].float()
            target_rms = torch.sqrt(torch.mean(target.detach().square()) + 1e-12)
            if float(target_rms) < self.silence_rms:
                losses.append(estimate.sum() * 0.0)
                continue
            if length < self.minimum_samples:
                # Zero padding creates a long synthetic silence boundary and
                # produces unstable IIR/PESQ gradients. Periodic extension is
                # deterministic, differentiable, and preserves the short
                # utterance's local signal statistics.
                repeats = (self.minimum_samples + length - 1) // length
                estimate = estimate.repeat(1, repeats)[:, : self.minimum_samples]
                target = target.repeat(1, repeats)[:, : self.minimum_samples]
            device_type = estimate.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                losses.append(self.loss(target, estimate).mean())
        return torch.stack(losses).mean()


@dataclass(frozen=True)
class T3LossBreakdown:
    total: torch.Tensor
    mrstft: torch.Tensor
    sisdr: torch.Tensor
    anchor: torch.Tensor
    pmsqe: torch.Tensor


class T3TeacherObjective(nn.Module):
    """Matched E1/E2 objective with frozen, externally calibrated weights."""

    def __init__(
        self,
        *,
        branch: str,
        anchor_weight: float,
        pmsqe_weight: float = 0.0,
        sample_rate: int = T3_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        normalized_branch = str(branch).strip().upper()
        if normalized_branch not in {"E1-SUP", "E2-PMSQE"}:
            raise ValueError("branch must be E1-SUP or E2-PMSQE.")
        if int(sample_rate) != T3_SAMPLE_RATE:
            raise ValueError("T3 teacher objectives require WB/16 kHz audio.")
        if anchor_weight < 0.0 or pmsqe_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if normalized_branch == "E1-SUP" and pmsqe_weight != 0.0:
            raise ValueError("E1-SUP cannot include the PMSQE term.")
        if normalized_branch == "E2-PMSQE" and pmsqe_weight <= 0.0:
            raise ValueError("E2-PMSQE requires a positive frozen PMSQE weight.")
        self.branch = normalized_branch
        self.sample_rate = int(sample_rate)
        self.anchor_weight = float(anchor_weight)
        self.pmsqe_weight = float(pmsqe_weight)
        self.mrstft = MultiResolutionSTFTLoss()
        self.sisdr = TrueLengthSISDRLoss()
        self.anchor = T0LogMagnitudeAnchorLoss()
        self.pmsqe = DifferentiablePESQInspiredLoss(sample_rate=sample_rate)

    def forward(
        self,
        candidate: torch.Tensor,
        clean: torch.Tensor,
        teacher_t0: torch.Tensor,
        *,
        lengths: torch.Tensor | Iterable[int] | None = None,
    ) -> T3LossBreakdown:
        candidate, clean = _validate_aligned(candidate, clean, reference_name="clean")
        _, teacher_t0 = _validate_aligned(
            candidate,
            teacher_t0,
            reference_name="teacher_t0",
        )
        mrstft = self.mrstft(candidate, clean, lengths=lengths)
        sisdr = self.sisdr(candidate, clean, lengths=lengths)
        anchor = self.anchor(candidate, teacher_t0, lengths=lengths)
        pmsqe = candidate.new_tensor(0.0, dtype=torch.float32)
        total = mrstft + 0.10 * sisdr + self.anchor_weight * anchor
        if self.branch == "E2-PMSQE":
            pmsqe = self.pmsqe(candidate, clean, lengths=lengths)
            total = total + self.pmsqe_weight * pmsqe
        return T3LossBreakdown(
            total=total,
            mrstft=mrstft,
            sisdr=sisdr,
            anchor=anchor,
            pmsqe=pmsqe,
        )


@dataclass(frozen=True)
class T3GradientCalibration:
    supervised_norm: float
    anchor_norm: float
    pmsqe_norm: float
    anchor_weight: float
    pmsqe_weight: float
    anchor_fraction_bound: float
    pmsqe_fraction_bound: float


def calibrate_t3_gradient_weights(
    *,
    candidate: torch.Tensor,
    clean: torch.Tensor,
    teacher_t0: torch.Tensor,
    lengths: torch.Tensor | Iterable[int] | None = None,
    anchor_fraction_bound: float = 0.50,
    pmsqe_fraction_bound: float = 0.10,
    eps: float = 1e-12,
) -> T3GradientCalibration:
    """Freeze weights from one train-only, non-zero local candidate.

    The caller must supply a teacher-manifold candidate distinct from T0; the
    exact T0 anchor gradient is zero and cannot identify an anchor scale.
    """

    if not candidate.requires_grad:
        raise ValueError("candidate must require gradients for calibration.")
    if not 0.0 < anchor_fraction_bound < 1.0:
        raise ValueError("anchor_fraction_bound must be in (0, 1).")
    if not 0.0 < pmsqe_fraction_bound < 1.0:
        raise ValueError("pmsqe_fraction_bound must be in (0, 1).")
    mrstft = MultiResolutionSTFTLoss().to(candidate.device)(
        candidate,
        clean,
        lengths=lengths,
    )
    sisdr = TrueLengthSISDRLoss().to(candidate.device)(
        candidate,
        clean,
        lengths=lengths,
    )
    anchor = T0LogMagnitudeAnchorLoss().to(candidate.device)(
        candidate,
        teacher_t0,
        lengths=lengths,
    )
    pmsqe = DifferentiablePESQInspiredLoss().to(candidate.device)(
        candidate,
        clean,
        lengths=lengths,
    )
    components = (mrstft + 0.10 * sisdr, anchor, pmsqe)
    norms: list[float] = []
    for component in components:
        gradient = torch.autograd.grad(
            component,
            candidate,
            retain_graph=True,
            create_graph=False,
        )[0]
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        if not torch.isfinite(gradient).all() or norm <= eps:
            raise RuntimeError("T3 gradient calibration found a non-finite or vanishing term.")
        norms.append(norm)
    supervised_norm, anchor_norm, pmsqe_norm = norms
    anchor_contribution = (
        anchor_fraction_bound / (1.0 - anchor_fraction_bound)
    ) * supervised_norm
    anchor_weight = anchor_contribution / anchor_norm
    pre_pmsqe_contribution = supervised_norm + anchor_contribution
    pmsqe_contribution = (
        pmsqe_fraction_bound / (1.0 - pmsqe_fraction_bound)
    ) * pre_pmsqe_contribution
    pmsqe_weight = pmsqe_contribution / pmsqe_norm
    return T3GradientCalibration(
        supervised_norm=supervised_norm,
        anchor_norm=anchor_norm,
        pmsqe_norm=pmsqe_norm,
        anchor_weight=anchor_weight,
        pmsqe_weight=pmsqe_weight,
        anchor_fraction_bound=float(anchor_fraction_bound),
        pmsqe_fraction_bound=float(pmsqe_fraction_bound),
    )
