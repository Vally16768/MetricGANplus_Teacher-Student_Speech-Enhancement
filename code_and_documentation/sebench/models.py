from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import huggingface_hub
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from sebench.postfilters import PostFilterEnhancer, resolve_postfilter_config
from sebench.bandwidth import resolve_bandwidth


MODEL_FAMILIES = (
    "metricgan_plus",
    "metricgan_plus_teacher_wb",
    "metricgan_plus_teacher_official_wb",
    "metricgan_plus_student_wb",
    "metricgan_plus_student_nb",
    "metricgan_plus_student_wb_causal_max",
    "metricgan_plus_student_nb_causal_max",
    "metricgan_plus_native8k",
    "metricgan_plus_native8k_causal_s",
    "metricgan_plus_native8k_causal_xs",
    "metricgan_plus_native8k_causal_n6",
    "metricgan_plus_native8k_causal_max",
)
MODEL_VARIANTS = ("small", "base")
DEFAULT_MICROBATCH = {
    "metricgan_plus": 8,
    "metricgan_plus_teacher_wb": 8,
    "metricgan_plus_teacher_official_wb": 8,
    "metricgan_plus_student_wb": 12,
    "metricgan_plus_student_nb": 12,
    "metricgan_plus_student_wb_causal_max": 8,
    "metricgan_plus_student_nb_causal_max": 8,
    "metricgan_plus_native8k": 8,
    "metricgan_plus_native8k_causal_s": 12,
    "metricgan_plus_native8k_causal_xs": 14,
    "metricgan_plus_native8k_causal_n6": 10,
    "metricgan_plus_native8k_causal_max": 8,
}
METRICGAN_PLUS_SOURCE = "speechbrain/metricgan-plus-voicebank"
METRICGAN_PLUS_CACHE_DIR = Path.home() / ".cache" / "sebench" / "metricgan_plus_voicebank"
METRICGAN_PLUS_HF_REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"
METRICGAN_PLUS_CHECKPOINT_SHA256 = (
    "147bfb866bac8264603546e035bf283370e716ed2f4b7412d308d2bcee88304f"
)


class WaveformEnhancer(nn.Module):
    def preferred_microbatch(self) -> int | None:
        configured = getattr(self, "eval_microbatch", None)
        if configured is not None:
            value = int(configured)
            return value if value > 0 else None
        model_config = getattr(self, "model_config", None)
        if isinstance(model_config, dict):
            arch = str(model_config.get("arch") or "").strip()
            if arch:
                value = int(DEFAULT_MICROBATCH.get(arch, 0) or 0)
                return value if value > 0 else None
        return None

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 2:
            raise ValueError("Expected noisy tensor shaped (batch, length).")
        microbatch = self.preferred_microbatch()
        if microbatch is not None and noisy.shape[0] > microbatch:
            chunks = []
            for start in range(0, noisy.shape[0], microbatch):
                stop = start + microbatch
                chunks.append(self.forward(noisy[start:stop].unsqueeze(1)))
            enhanced = torch.cat(chunks, dim=0)
        else:
            enhanced = self.forward(noisy.unsqueeze(1))
        if enhanced.ndim != 3:
            raise ValueError("Model forward must return shape (batch, 1, length).")
        return enhanced.squeeze(1)


class MetricGANPlusAdapter(WaveformEnhancer):
    _bundle_cache: dict[str, object] = {}

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)
        self.model_config = {
            "arch": "metricgan_plus",
            "variant": variant,
            "sample_rate": 16000,
            "non_causal": True,
            "frozen_pretrained": True,
        }

    @staticmethod
    def _device_string(device: torch.device) -> str:
        if device.type != "cuda":
            return device.type
        return f"cuda:{device.index}" if device.index is not None else "cuda"

    @classmethod
    def _bundle_for_device(cls, device: torch.device) -> object:
        device_str = cls._device_string(device)
        bundle = cls._bundle_cache.get(device_str)
        if bundle is None or cls._bundle_has_inference_tensors(bundle):
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["ffmpeg"]  # type: ignore[attr-defined]
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]
            if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
                original_hf_hub_download = huggingface_hub.hf_hub_download

                def _hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
                    if use_auth_token is not None and "token" not in kwargs:
                        kwargs["token"] = use_auth_token
                    try:
                        return original_hf_hub_download(*args, **kwargs)
                    except Exception as exc:
                        if exc.__class__.__name__ == "RemoteEntryNotFoundError":
                            raise ValueError("File not found on HF hub") from exc
                        raise

                huggingface_hub.hf_hub_download = _hf_hub_download_compat
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
                from speechbrain.utils.fetching import FetchConfig
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "MetricGAN+ support requires SpeechBrain. Install `speechbrain>=1.0.0`."
                ) from exc
            METRICGAN_PLUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # SpeechBrain may build cached modules while the caller is in inference_mode.
            # Force normal tensors here so the frozen stage-1 can be reused during training.
            with torch.inference_mode(False):
                bundle = SpectralMaskEnhancement.from_hparams(
                    source=METRICGAN_PLUS_SOURCE,
                    savedir=str(METRICGAN_PLUS_CACHE_DIR),
                    run_opts={"device": device_str},
                    fetch_config=FetchConfig(
                        revision=METRICGAN_PLUS_HF_REVISION,
                    ),
                )
            bundle.eval()
            cls._bundle_cache[device_str] = bundle
        return cls._bundle_cache[device_str]

    @staticmethod
    def _bundle_has_inference_tensors(bundle: object) -> bool:
        modules = getattr(bundle, "mods", None)
        if modules is None:
            return False
        tensors = list(modules.parameters()) + list(modules.buffers())
        return any(getattr(tensor, "is_inference", lambda: False)() for tensor in tensors)

    @classmethod
    def pretrained_generator_state_dict(cls, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        bundle = cls._bundle_for_device(torch.device(device))
        checkpoint_path = METRICGAN_PLUS_CACHE_DIR / "enhance_model.ckpt"
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != METRICGAN_PLUS_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Official MetricGAN+ checkpoint hash mismatch: expected "
                f"{METRICGAN_PLUS_CHECKPOINT_SHA256}, got {digest}."
            )
        enhance_model = bundle.mods["enhance_model"]
        return {key: value.detach().cpu().clone() for key, value in enhance_model.state_dict().items()}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        bundle = self._bundle_for_device(input.device)
        noisy = input.squeeze(1)
        lengths = torch.ones(noisy.size(0), device=noisy.device, dtype=torch.float32)
        with torch.no_grad():
            enhanced = bundle.enhance_batch(noisy, lengths=lengths)
        if isinstance(enhanced, tuple):
            enhanced = enhanced[0]
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        if enhanced.ndim == 3 and enhanced.shape[-1] == 1:
            enhanced = enhanced.transpose(1, 2)
        if enhanced.ndim == 2:
            enhanced = enhanced.unsqueeze(1)
        if enhanced.ndim != 3 or enhanced.shape[1] != 1:
            raise ValueError(f"Unexpected MetricGAN+ output shape: {tuple(enhanced.shape)}")
        enhanced = enhanced.to(input.device)
        return enhanced[..., : input.shape[-1]].clone()


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 1.2 * torch.sigmoid(self.slope * features)


def _fake_quant_tensor(tensor: torch.Tensor, enabled: bool, num_bits: int = 8) -> torch.Tensor:
    if not enabled or not tensor.is_floating_point():
        return tensor
    levels = float(2**num_bits - 1)
    max_val = tensor.detach().abs().max()
    if float(max_val) < 1e-8:
        return tensor
    scale = max_val / (levels / 2.0)
    quantized = torch.clamp(torch.round(tensor / scale), min=-(levels / 2.0), max=levels / 2.0)
    dequantized = quantized * scale
    return tensor + (dequantized - tensor).detach()


class MetricGANLikeMaskGenerator(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        linear_dim: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        for name, param in self.blstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
        self.linear1 = nn.Linear(hidden_size * 2, linear_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        self.linear2 = nn.Linear(linear_dim, output_size)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        self.learnable_sigmoid = LearnableSigmoid(output_size)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.blstm(features)
        encoded = self.activation(self.linear1(encoded))
        return self.linear2(encoded)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.learnable_sigmoid(self.forward_logits(features))


class MetricGANCausalLiteMaskGenerator(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        linear_dim: int,
        output_size: int,
        rnn_type: str = "gru",
        qat: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.qat = qat
        if self.rnn_type == "gru":
            self.rnn: nn.Module = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unsupported causal MetricGAN RNN type: {rnn_type}")
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
        self.linear1 = nn.Linear(hidden_size, linear_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        self.linear2 = nn.Linear(linear_dim, output_size)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        self.learnable_sigmoid = LearnableSigmoid(output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = _fake_quant_tensor(features, self.qat)
        encoded, _ = self.rnn(features)
        encoded = _fake_quant_tensor(encoded, self.qat)
        encoded = self.activation(self.linear1(encoded))
        encoded = _fake_quant_tensor(encoded, self.qat)
        encoded = self.linear2(encoded)
        encoded = _fake_quant_tensor(encoded, self.qat)
        return self.learnable_sigmoid(encoded)


class MetricGANLikeEnhancer(WaveformEnhancer):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        hidden_size: int,
        num_layers: int,
        linear_dim: int,
        arch_name: str,
        init_from_pretrained: bool,
        feature_domain: str = "sqrt_magnitude",
        window_type: str = "hann",
        official_checkpoint_sha256: str | None = None,
        confidence_calibration_enabled: bool = False,
        confidence_calibration_low: float = 0.0,
        confidence_calibration_high: float = 0.0,
        confidence_calibration_threshold: float = -4.0,
        confidence_calibration_temperature: float = 1.5,
        adaptive_router_enabled: bool = False,
        adaptive_router_feature_mean: list[float] | tuple[float, ...] | None = None,
        adaptive_router_feature_scale: list[float] | tuple[float, ...] | None = None,
        adaptive_router_weights: list[float] | tuple[float, ...] | None = None,
        adaptive_router_bias: float = 0.0,
        adaptive_router_threshold: float = 0.0,
        multi_router_enabled: bool = False,
        multi_router_lows: list[float] | tuple[float, ...] | None = None,
        multi_router_feature_means: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        multi_router_feature_scales: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        multi_router_weights: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        multi_router_biases: list[float] | tuple[float, ...] | None = None,
        multi_router_threshold: float = 0.0,
        multi_router_feature_transform: str = "identity",
    ) -> None:
        super().__init__()
        feature_bins = n_fft // 2 + 1
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.feature_bins = feature_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_dim = linear_dim
        self.feature_domain = str(feature_domain)
        self.confidence_calibration_enabled = bool(
            confidence_calibration_enabled
        )
        self.confidence_calibration_low = float(confidence_calibration_low)
        self.confidence_calibration_high = float(confidence_calibration_high)
        self.confidence_calibration_threshold = float(
            confidence_calibration_threshold
        )
        self.confidence_calibration_temperature = float(
            confidence_calibration_temperature
        )
        self.adaptive_router_enabled = bool(adaptive_router_enabled)
        self.adaptive_router_feature_mean = tuple(
            float(value) for value in (adaptive_router_feature_mean or ())
        )
        self.adaptive_router_feature_scale = tuple(
            float(value) for value in (adaptive_router_feature_scale or ())
        )
        self.adaptive_router_weights = tuple(
            float(value) for value in (adaptive_router_weights or ())
        )
        self.adaptive_router_bias = float(adaptive_router_bias)
        self.adaptive_router_threshold = float(adaptive_router_threshold)
        self.multi_router_enabled = bool(multi_router_enabled)
        self.multi_router_lows = tuple(float(value) for value in (multi_router_lows or ()))
        self.multi_router_feature_means = tuple(
            tuple(float(value) for value in row)
            for row in (multi_router_feature_means or ())
        )
        self.multi_router_feature_scales = tuple(
            tuple(float(value) for value in row)
            for row in (multi_router_feature_scales or ())
        )
        self.multi_router_weights = tuple(
            tuple(float(value) for value in row)
            for row in (multi_router_weights or ())
        )
        self.multi_router_biases = tuple(
            float(value) for value in (multi_router_biases or ())
        )
        self.multi_router_threshold = float(multi_router_threshold)
        self.multi_router_feature_transform = str(multi_router_feature_transform)
        if self.confidence_calibration_temperature <= 0.0:
            raise ValueError(
                "Confidence-calibration temperature must be positive."
            )
        if self.adaptive_router_enabled and not (
            len(self.adaptive_router_feature_mean)
            == len(self.adaptive_router_feature_scale)
            == len(self.adaptive_router_weights)
            == 16
        ):
            raise ValueError("Adaptive T8 router requires exactly 16 features.")
        if any(value <= 0.0 for value in self.adaptive_router_feature_scale):
            raise ValueError("Adaptive T8 router feature scales must be positive.")
        self._validate_multi_router()
        if self.feature_domain not in {
            "sqrt_magnitude",
            "official_log_magnitude",
        }:
            raise ValueError(f"Unsupported MetricGAN feature domain: {feature_domain}")
        self.window_type = str(window_type)
        if self.window_type == "hann":
            window = torch.hann_window(win_length)
        elif self.window_type == "hamming":
            window = torch.hamming_window(win_length)
        else:
            raise ValueError(f"Unsupported MetricGAN window type: {window_type}")
        self.register_buffer("window", window, persistent=False)
        self.mask_generator = MetricGANLikeMaskGenerator(
            input_size=feature_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            linear_dim=linear_dim,
            output_size=feature_bins,
        )
        self.model_config = {
            "arch": arch_name,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "feature_bins": feature_bins,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": True,
            "linear_dims": [hidden_size * 2, linear_dim, feature_bins],
            "sequence_frames": 100,
            "non_causal": True,
            "lookahead_ms": 500.0,
            "init_from_metricgan_pretrained": init_from_pretrained,
            "feature_domain": self.feature_domain,
            "window_type": self.window_type,
            # Reconstruct saved packages without requiring a network/cache hit.
            "initialize_from_official": False,
            "confidence_calibration_enabled": self.confidence_calibration_enabled,
            "confidence_calibration_low": self.confidence_calibration_low,
            "confidence_calibration_high": self.confidence_calibration_high,
            "confidence_calibration_threshold": self.confidence_calibration_threshold,
            "confidence_calibration_temperature": self.confidence_calibration_temperature,
            "adaptive_router_enabled": self.adaptive_router_enabled,
            "adaptive_router_feature_mean": list(self.adaptive_router_feature_mean),
            "adaptive_router_feature_scale": list(self.adaptive_router_feature_scale),
            "adaptive_router_weights": list(self.adaptive_router_weights),
            "adaptive_router_bias": self.adaptive_router_bias,
            "adaptive_router_threshold": self.adaptive_router_threshold,
            "multi_router_enabled": self.multi_router_enabled,
            "multi_router_lows": list(self.multi_router_lows),
            "multi_router_feature_means": [
                list(row) for row in self.multi_router_feature_means
            ],
            "multi_router_feature_scales": [
                list(row) for row in self.multi_router_feature_scales
            ],
            "multi_router_weights": [list(row) for row in self.multi_router_weights],
            "multi_router_biases": list(self.multi_router_biases),
            "multi_router_threshold": self.multi_router_threshold,
            "multi_router_feature_transform": self.multi_router_feature_transform,
        }
        if official_checkpoint_sha256:
            self.model_config["official_checkpoint_sha256"] = str(
                official_checkpoint_sha256
            )
            self.model_config["official_source"] = METRICGAN_PLUS_SOURCE
            self.model_config["official_revision"] = METRICGAN_PLUS_HF_REVISION
        self.pretrained_init_summary = {
            "loaded_keys": [],
            "skipped_keys": [],
            "loaded_key_count": 0,
            "skipped_key_count": 0,
        }
        if init_from_pretrained:
            self._init_from_metricgan_pretrained()

    def _init_from_metricgan_pretrained(self) -> None:
        pretrained = MetricGANPlusAdapter.pretrained_generator_state_dict("cpu")
        remapped = {
            key.replace("blstm.rnn.", "mask_generator.blstm.")
            .replace("linear1.", "mask_generator.linear1.")
            .replace("linear2.", "mask_generator.linear2.")
            .replace("Learnable_sigmoid.", "mask_generator.learnable_sigmoid."): value
            for key, value in pretrained.items()
        }
        current_state = self.state_dict()
        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        for key, value in remapped.items():
            target = current_state.get(key)
            if target is not None and tuple(target.shape) == tuple(value.shape):
                current_state[key] = value.to(dtype=target.dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        self.load_state_dict(current_state, strict=False)
        self.pretrained_init_summary = {
            "loaded_keys": loaded_keys,
            "skipped_keys": skipped_keys,
            "loaded_key_count": len(loaded_keys),
            "skipped_key_count": len(skipped_keys),
        }
        self.model_config["pretrained_init_summary"] = {
            "loaded_key_count": len(loaded_keys),
            "skipped_key_count": len(skipped_keys),
        }

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=spec.device, dtype=spec.real.dtype),
            center=True,
            length=length,
        ).unsqueeze(1)

    def configure_confidence_calibration(
        self,
        *,
        enabled: bool,
        low: float,
        high: float,
        threshold: float,
        temperature: float,
    ) -> None:
        """Configure the deployable T7 confidence-conditioned logit transform."""
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError(
                "Confidence-calibration temperature must be positive."
            )
        self.confidence_calibration_enabled = bool(enabled)
        self.confidence_calibration_low = float(low)
        self.confidence_calibration_high = float(high)
        self.confidence_calibration_threshold = float(threshold)
        self.confidence_calibration_temperature = temperature
        self.model_config.update(
            {
                "confidence_calibration_enabled": self.confidence_calibration_enabled,
                "confidence_calibration_low": self.confidence_calibration_low,
                "confidence_calibration_high": self.confidence_calibration_high,
                "confidence_calibration_threshold": self.confidence_calibration_threshold,
                "confidence_calibration_temperature": self.confidence_calibration_temperature,
            }
        )

    def calibrate_mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply T7 calibration without a clean-reference dependency."""
        if not self.confidence_calibration_enabled:
            return logits
        return self._confidence_candidate_logits(logits)

    def _confidence_candidate_logits(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        return self._confidence_candidate_logits_for(
            logits,
            low=self.confidence_calibration_low,
            high=self.confidence_calibration_high,
            threshold=self.confidence_calibration_threshold,
            temperature=self.confidence_calibration_temperature,
        )

    @staticmethod
    def _confidence_candidate_logits_for(
        logits: torch.Tensor,
        *,
        low: float,
        high: float,
        threshold: float,
        temperature: float,
    ) -> torch.Tensor:
        if float(temperature) <= 0.0:
            raise ValueError("Confidence-calibration temperature must be positive.")
        gate = torch.sigmoid(
            (logits - float(threshold)) / float(temperature)
        )
        correction = float(low) + (float(high) - float(low)) * gate
        return logits + correction

    def _validate_multi_router(self) -> None:
        if self.multi_router_feature_transform not in {"identity", "quadratic"}:
            raise ValueError("Unsupported multi-router feature transform.")
        expected_features = (
            16 if self.multi_router_feature_transform == "identity" else 152
        )
        action_count = len(self.multi_router_lows)
        collections = (
            self.multi_router_feature_means,
            self.multi_router_feature_scales,
            self.multi_router_weights,
            self.multi_router_biases,
        )
        if self.multi_router_enabled and (
            action_count < 2 or any(len(values) != action_count for values in collections)
        ):
            raise ValueError("T9 multi-action router requires at least two aligned actions.")
        for rows in (
            self.multi_router_feature_means,
            self.multi_router_feature_scales,
            self.multi_router_weights,
        ):
            if any(len(row) != expected_features for row in rows):
                raise ValueError(
                    "Multi-action router feature dimension does not match transform."
                )
        if any(
            value <= 0.0
            for row in self.multi_router_feature_scales
            for value in row
        ):
            raise ValueError("T9 multi-action router feature scales must be positive.")

    def configure_multi_action_router(
        self,
        *,
        enabled: bool,
        lows: list[float] | tuple[float, ...],
        feature_means: list[list[float]] | tuple[tuple[float, ...], ...],
        feature_scales: list[list[float]] | tuple[tuple[float, ...], ...],
        weights: list[list[float]] | tuple[tuple[float, ...], ...],
        biases: list[float] | tuple[float, ...],
        threshold: float,
        feature_transform: str = "identity",
    ) -> None:
        self.multi_router_enabled = bool(enabled)
        self.multi_router_lows = tuple(float(value) for value in lows)
        self.multi_router_feature_means = tuple(
            tuple(float(value) for value in row) for row in feature_means
        )
        self.multi_router_feature_scales = tuple(
            tuple(float(value) for value in row) for row in feature_scales
        )
        self.multi_router_weights = tuple(
            tuple(float(value) for value in row) for row in weights
        )
        self.multi_router_biases = tuple(float(value) for value in biases)
        self.multi_router_threshold = float(threshold)
        self.multi_router_feature_transform = str(feature_transform)
        self._validate_multi_router()
        self.model_config.update(
            {
                "multi_router_enabled": self.multi_router_enabled,
                "multi_router_lows": list(self.multi_router_lows),
                "multi_router_feature_means": [
                    list(row) for row in self.multi_router_feature_means
                ],
                "multi_router_feature_scales": [
                    list(row) for row in self.multi_router_feature_scales
                ],
                "multi_router_weights": [
                    list(row) for row in self.multi_router_weights
                ],
                "multi_router_biases": list(self.multi_router_biases),
                "multi_router_threshold": self.multi_router_threshold,
                "multi_router_feature_transform": self.multi_router_feature_transform,
            }
        )

    def configure_adaptive_router(
        self,
        *,
        enabled: bool,
        feature_mean: list[float] | tuple[float, ...],
        feature_scale: list[float] | tuple[float, ...],
        weights: list[float] | tuple[float, ...],
        bias: float,
        threshold: float,
    ) -> None:
        mean = tuple(float(value) for value in feature_mean)
        scale = tuple(float(value) for value in feature_scale)
        router_weights = tuple(float(value) for value in weights)
        if bool(enabled) and not (len(mean) == len(scale) == len(router_weights) == 16):
            raise ValueError("Adaptive T8 router requires exactly 16 features.")
        if any(value <= 0.0 for value in scale):
            raise ValueError("Adaptive T8 router feature scales must be positive.")
        self.adaptive_router_enabled = bool(enabled)
        self.adaptive_router_feature_mean = mean
        self.adaptive_router_feature_scale = scale
        self.adaptive_router_weights = router_weights
        self.adaptive_router_bias = float(bias)
        self.adaptive_router_threshold = float(threshold)
        self.model_config.update(
            {
                "adaptive_router_enabled": self.adaptive_router_enabled,
                "adaptive_router_feature_mean": list(mean),
                "adaptive_router_feature_scale": list(scale),
                "adaptive_router_weights": list(router_weights),
                "adaptive_router_bias": self.adaptive_router_bias,
                "adaptive_router_threshold": self.adaptive_router_threshold,
            }
        )

    def confidence_router_features(
        self,
        noisy: torch.Tensor,
        magnitude: torch.Tensor,
        logits: torch.Tensor,
        base_mask: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the frozen 16-feature clean-free T8 router schema."""
        batch = logits.shape[0]

        def flat(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, -1).float()

        noisy_flat = flat(noisy)
        magnitude_flat = flat(magnitude)
        logits_flat = flat(logits)
        mask_flat = flat(base_mask)
        correction_flat = flat(candidate_logits - logits)
        disagreement_flat = flat(candidate_mask - base_mask)
        log_rms = torch.log(
            noisy_flat.square().mean(dim=1).clamp_min(1e-12)
        ) * 0.5
        log_magnitude = torch.log1p(magnitude_flat)
        frequency = torch.linspace(
            0.0,
            1.0,
            magnitude.shape[1],
            device=magnitude.device,
            dtype=magnitude.dtype,
        ).reshape(1, -1, 1)
        centroid = (
            (magnitude * frequency).sum(dim=(1, 2))
            / magnitude.sum(dim=(1, 2)).clamp_min(1e-12)
        ).float()
        quantiles = torch.quantile(
            logits_flat,
            torch.tensor(
                [0.25, 0.50, 0.75],
                device=logits_flat.device,
                dtype=logits_flat.dtype,
            ),
            dim=1,
        ).transpose(0, 1)
        return torch.stack(
            (
                log_rms,
                log_magnitude.mean(dim=1),
                log_magnitude.std(dim=1, unbiased=False),
                centroid,
                logits_flat.mean(dim=1),
                logits_flat.std(dim=1, unbiased=False),
                quantiles[:, 0],
                quantiles[:, 1],
                quantiles[:, 2],
                mask_flat.mean(dim=1),
                mask_flat.std(dim=1, unbiased=False),
                (mask_flat < 0.25).float().mean(dim=1),
                (mask_flat > 0.75).float().mean(dim=1),
                correction_flat.mean(dim=1),
                correction_flat.std(dim=1, unbiased=False),
                disagreement_flat.abs().mean(dim=1),
            ),
            dim=1,
        )

    def adaptive_router_scores(self, features: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_router_enabled:
            raise RuntimeError("Adaptive T8 router is not enabled.")
        mean = features.new_tensor(self.adaptive_router_feature_mean)
        scale = features.new_tensor(self.adaptive_router_feature_scale)
        weights = features.new_tensor(self.adaptive_router_weights)
        normalized = (features - mean) / scale
        return normalized.matmul(weights) + self.adaptive_router_bias

    def multi_router_scores(
        self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if not self.multi_router_enabled or len(features) != len(self.multi_router_lows):
            raise RuntimeError("T9 multi-action router is not enabled or is incomplete.")
        scores = []
        for index, action_features in enumerate(features):
            if self.multi_router_feature_transform == "quadratic":
                products = [
                    action_features[:, left] * action_features[:, right]
                    for left in range(16)
                    for right in range(left, 16)
                ]
                action_features = torch.cat(
                    (action_features, torch.stack(products, dim=1)),
                    dim=1,
                )
            mean = action_features.new_tensor(self.multi_router_feature_means[index])
            scale = action_features.new_tensor(self.multi_router_feature_scales[index])
            weights = action_features.new_tensor(self.multi_router_weights[index])
            normalized = (action_features - mean) / scale
            scores.append(
                normalized.matmul(weights) + self.multi_router_biases[index]
            )
        return torch.stack(scores, dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        original_length = input.shape[-1]
        spec = self._stft(input)
        magnitude = spec.abs().clamp_min(1e-8)
        if self.feature_domain == "official_log_magnitude":
            features = torch.log1p(magnitude)
        else:
            features = magnitude.pow(0.5)
        features = features.transpose(1, 2)
        if self.multi_router_enabled:
            logits = self.mask_generator.forward_logits(features)
            base_mask = self.mask_generator.learnable_sigmoid(logits)
            action_masks: list[torch.Tensor] = []
            action_features: list[torch.Tensor] = []
            for low in self.multi_router_lows:
                candidate_logits = self._confidence_candidate_logits_for(
                    logits,
                    low=low,
                    high=0.0,
                    threshold=0.0,
                    temperature=1.5,
                )
                candidate_mask = self.mask_generator.learnable_sigmoid(candidate_logits)
                action_masks.append(candidate_mask)
                action_features.append(
                    self.confidence_router_features(
                        input,
                        magnitude,
                        logits,
                        base_mask,
                        candidate_logits,
                        candidate_mask,
                    )
                )
            scores = self.multi_router_scores(action_features)
            best_scores, best_actions = scores.max(dim=1)
            stacked_masks = torch.stack(action_masks, dim=1)
            gather_index = best_actions.reshape(-1, 1, 1, 1).expand(
                -1, 1, stacked_masks.shape[2], stacked_masks.shape[3]
            )
            selected_mask = stacked_masks.gather(1, gather_index).squeeze(1)
            use_action = (best_scores >= self.multi_router_threshold).reshape(-1, 1, 1)
            mask = torch.where(use_action, selected_mask, base_mask)
        elif self.adaptive_router_enabled:
            logits = self.mask_generator.forward_logits(features)
            base_mask = self.mask_generator.learnable_sigmoid(logits)
            candidate_logits = self._confidence_candidate_logits(logits)
            candidate_mask = self.mask_generator.learnable_sigmoid(candidate_logits)
            router_features = self.confidence_router_features(
                input,
                magnitude,
                logits,
                base_mask,
                candidate_logits,
                candidate_mask,
            )
            use_candidate = (
                self.adaptive_router_scores(router_features)
                >= self.adaptive_router_threshold
            ).reshape(-1, 1, 1)
            mask = torch.where(use_candidate, candidate_mask, base_mask)
        elif self.confidence_calibration_enabled:
            logits = self.mask_generator.forward_logits(features)
            logits = self.calibrate_mask_logits(logits)
            mask = self.mask_generator.learnable_sigmoid(logits)
        else:
            mask = self.mask_generator(features)
        mask = mask.transpose(1, 2).clamp_min(0.0)
        masked = mask * features.transpose(1, 2)
        if self.feature_domain == "official_log_magnitude":
            enhanced_magnitude = torch.expm1(masked).clamp_min(0.0)
        else:
            enhanced_magnitude = masked.pow(2.0)
        enhanced_spec = torch.polar(enhanced_magnitude, torch.angle(spec))
        enhanced = self._istft(enhanced_spec, original_length)
        return enhanced[..., :original_length]

    def forward_mask_logit_variants(
        self,
        input: torch.Tensor,
        logit_deltas: tuple[float, ...],
    ) -> torch.Tensor:
        """Return teacher-manifold candidates from bounded mask-logit shifts.

        The output shape is ``[variants, batch, 1, time]``. A zero delta is
        exactly the ordinary forward path and provides a parity fixture.
        """
        if input.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        if not logit_deltas:
            raise ValueError("At least one mask-logit delta is required.")
        if any(abs(float(delta)) > 0.10 for delta in logit_deltas):
            raise ValueError("T3 mask-logit perturbations are bounded to +/-0.10.")
        original_length = input.shape[-1]
        spec = self._stft(input)
        magnitude = spec.abs().clamp_min(1e-8)
        if self.feature_domain == "official_log_magnitude":
            features_frequency_first = torch.log1p(magnitude)
        else:
            features_frequency_first = magnitude.pow(0.5)
        features = features_frequency_first.transpose(1, 2)
        logits = self.mask_generator.forward_logits(features)
        logits = self.calibrate_mask_logits(logits)
        outputs: list[torch.Tensor] = []
        for delta in logit_deltas:
            mask = self.mask_generator.learnable_sigmoid(
                logits + float(delta)
            ).transpose(1, 2).clamp_min(0.0)
            masked = mask * features_frequency_first
            if self.feature_domain == "official_log_magnitude":
                enhanced_magnitude = torch.expm1(masked).clamp_min(0.0)
            else:
                enhanced_magnitude = masked.pow(2.0)
            enhanced_spec = torch.polar(enhanced_magnitude, torch.angle(spec))
            enhanced = self._istft(enhanced_spec, original_length)
            outputs.append(enhanced[..., :original_length])
        return torch.stack(outputs, dim=0)

    def forward_with_mask_logit_delta(
        self,
        input: torch.Tensor,
        logit_delta: float,
    ) -> torch.Tensor:
        return self.forward_mask_logit_variants(
            input,
            (float(logit_delta),),
        )[0]


class MetricGANCausalLiteEnhancer(WaveformEnhancer):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        hidden_size: int,
        num_layers: int,
        linear_dim: int,
        arch_name: str,
        rnn_type: str = "gru",
        qat: bool = False,
    ) -> None:
        super().__init__()
        feature_bins = n_fft // 2 + 1
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.feature_bins = feature_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_dim = linear_dim
        self.rnn_type = rnn_type.lower()
        self.qat = qat
        self.register_buffer("window", torch.hamming_window(win_length), persistent=False)
        self.mask_generator = MetricGANCausalLiteMaskGenerator(
            input_size=feature_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            linear_dim=linear_dim,
            output_size=feature_bins,
            rnn_type=self.rnn_type,
            qat=qat,
        )
        lookahead_ms = float(n_fft // 2) / float(sample_rate) * 1000.0
        self.model_config = {
            "arch": arch_name,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "feature_bins": feature_bins,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": False,
            "rnn_type": self.rnn_type,
            "linear_dims": [hidden_size, linear_dim, feature_bins],
            "sequence_frames": 8,
            "non_causal": False,
            "lookahead_ms": lookahead_ms,
            "qat": qat,
        }

    def _pad_input(self, wav: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        original_length = wav.shape[-1]
        padded_length = max(original_length, self.win_length)
        remainder = (padded_length - self.win_length) % self.hop_length
        if remainder:
            padded_length += self.hop_length - remainder
        pad = padded_length - original_length
        if pad:
            wav = F.pad(wav, (0, pad))
        return wav, pad, padded_length

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=spec.device, dtype=spec.real.dtype),
            center=True,
            length=length,
        ).unsqueeze(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        padded_input, _, padded_length = self._pad_input(input)
        spec = self._stft(padded_input)
        magnitude = spec.abs().clamp_min(1e-8).pow(0.5)
        features = magnitude.transpose(1, 2)
        mask = self.mask_generator(features).transpose(1, 2).clamp_min(0.0)
        enhanced_magnitude = (mask * magnitude).pow(2.0)
        enhanced_spec = torch.polar(enhanced_magnitude, torch.angle(spec))
        enhanced = self._istft(enhanced_spec, padded_length)
        return enhanced[..., : input.shape[-1]]


def build_metricgan_standalone(
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    variant: str = "small",
    native8k: bool = False,
    init_from_pretrained: bool = True,
    arch_name: str | None = None,
    feature_domain: str = "sqrt_magnitude",
    window_type: str = "hann",
    official_checkpoint_sha256: str | None = None,
    confidence_calibration_enabled: bool = False,
    confidence_calibration_low: float = 0.0,
    confidence_calibration_high: float = 0.0,
    confidence_calibration_threshold: float = -4.0,
    confidence_calibration_temperature: float = 1.5,
    adaptive_router_enabled: bool = False,
    adaptive_router_feature_mean: list[float] | tuple[float, ...] | None = None,
    adaptive_router_feature_scale: list[float] | tuple[float, ...] | None = None,
    adaptive_router_weights: list[float] | tuple[float, ...] | None = None,
    adaptive_router_bias: float = 0.0,
    adaptive_router_threshold: float = 0.0,
    multi_router_enabled: bool = False,
    multi_router_lows: list[float] | tuple[float, ...] | None = None,
    multi_router_feature_means: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_feature_scales: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_weights: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_biases: list[float] | tuple[float, ...] | None = None,
    multi_router_threshold: float = 0.0,
    multi_router_feature_transform: str = "identity",
) -> MetricGANLikeEnhancer:
    if variant == "small":
        hidden_size = 200
        linear_dim = 300
    else:
        hidden_size = 256
        linear_dim = 384
    resolved_arch_name = (
        arch_name
        or ("metricgan_plus_native8k" if native8k else "metricgan_plus")
    )
    return MetricGANLikeEnhancer(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        hidden_size=hidden_size,
        num_layers=2,
        linear_dim=linear_dim,
        arch_name=resolved_arch_name,
        init_from_pretrained=init_from_pretrained,
        feature_domain=feature_domain,
        window_type=window_type,
        official_checkpoint_sha256=official_checkpoint_sha256,
        confidence_calibration_enabled=confidence_calibration_enabled,
        confidence_calibration_low=confidence_calibration_low,
        confidence_calibration_high=confidence_calibration_high,
        confidence_calibration_threshold=confidence_calibration_threshold,
        confidence_calibration_temperature=confidence_calibration_temperature,
        adaptive_router_enabled=adaptive_router_enabled,
        adaptive_router_feature_mean=adaptive_router_feature_mean,
        adaptive_router_feature_scale=adaptive_router_feature_scale,
        adaptive_router_weights=adaptive_router_weights,
        adaptive_router_bias=adaptive_router_bias,
        adaptive_router_threshold=adaptive_router_threshold,
        multi_router_enabled=multi_router_enabled,
        multi_router_lows=multi_router_lows,
        multi_router_feature_means=multi_router_feature_means,
        multi_router_feature_scales=multi_router_feature_scales,
        multi_router_weights=multi_router_weights,
        multi_router_biases=multi_router_biases,
        multi_router_threshold=multi_router_threshold,
        multi_router_feature_transform=multi_router_feature_transform,
    )


def build_metricgan_causal_lite(
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    family: str,
    qat: bool = False,
) -> MetricGANCausalLiteEnhancer:
    configs = {
        "metricgan_plus_student_wb": {"hidden_size": 96, "num_layers": 1, "linear_dim": 128, "rnn_type": "gru"},
        "metricgan_plus_student_nb": {"hidden_size": 96, "num_layers": 1, "linear_dim": 128, "rnn_type": "gru"},
        "metricgan_plus_student_wb_causal_max": {
            "hidden_size": 160,
            "num_layers": 3,
            "linear_dim": 224,
            "rnn_type": "gru",
        },
        "metricgan_plus_student_nb_causal_max": {
            "hidden_size": 160,
            "num_layers": 3,
            "linear_dim": 224,
            "rnn_type": "gru",
        },
        "metricgan_plus_native8k_causal_s": {"hidden_size": 96, "num_layers": 1, "linear_dim": 128, "rnn_type": "gru"},
        "metricgan_plus_native8k_causal_xs": {"hidden_size": 64, "num_layers": 1, "linear_dim": 96, "rnn_type": "gru"},
        "metricgan_plus_native8k_causal_n6": {"hidden_size": 128, "num_layers": 2, "linear_dim": 160, "rnn_type": "gru"},
        "metricgan_plus_native8k_causal_max": {"hidden_size": 160, "num_layers": 3, "linear_dim": 224, "rnn_type": "gru"},
    }
    if family not in configs:
        raise ValueError(f"Unsupported causal MetricGAN family: {family}")
    cfg = configs[family]
    return MetricGANCausalLiteEnhancer(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        linear_dim=cfg["linear_dim"],
        arch_name=family,
        rnn_type=str(cfg["rnn_type"]),
        qat=qat,
    )


def dynamic_quantize_metricgan(model: nn.Module) -> nn.Module:
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized


def build_model(
    model_family: str,
    variant: str = "base",
    *,
    spectral_native_gate: bool = False,
    erb_bands: int = 32,
    context_frames: int = 5,
    guidance_classic: str = "none",
    qat: bool = False,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 320,
    initialize_from_official: bool = True,
    official_checkpoint_sha256: str | None = None,
    confidence_calibration_enabled: bool = False,
    confidence_calibration_low: float = 0.0,
    confidence_calibration_high: float = 0.0,
    confidence_calibration_threshold: float = -4.0,
    confidence_calibration_temperature: float = 1.5,
    adaptive_router_enabled: bool = False,
    adaptive_router_feature_mean: list[float] | tuple[float, ...] | None = None,
    adaptive_router_feature_scale: list[float] | tuple[float, ...] | None = None,
    adaptive_router_weights: list[float] | tuple[float, ...] | None = None,
    adaptive_router_bias: float = 0.0,
    adaptive_router_threshold: float = 0.0,
    multi_router_enabled: bool = False,
    multi_router_lows: list[float] | tuple[float, ...] | None = None,
    multi_router_feature_means: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_feature_scales: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_weights: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_biases: list[float] | tuple[float, ...] | None = None,
    multi_router_threshold: float = 0.0,
    multi_router_feature_transform: str = "identity",
) -> nn.Module:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model variant: {variant}")

    model_family = model_family.lower()
    if model_family == "metricgan_plus":
        if spectral_native_gate:
            raise ValueError("MetricGAN+ does not support spectral-native gating.")
        return MetricGANPlusAdapter(variant)
    if model_family == "metricgan_plus_teacher_wb":
        if spectral_native_gate:
            raise ValueError("MetricGAN+ WB teacher does not support spectral-native gating.")
        resolve_bandwidth("wb", sample_rate=sample_rate)
        return build_metricgan_standalone(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            variant=variant,
            native8k=True,
            init_from_pretrained=False,
            arch_name=model_family,
        )
    if model_family == "metricgan_plus_teacher_official_wb":
        if spectral_native_gate:
            raise ValueError(
                "Official MetricGAN+ WB teacher does not support spectral-native gating."
            )
        resolve_bandwidth("wb", sample_rate=sample_rate)
        if variant != "small":
            raise ValueError(
                "The official MetricGAN+ checkpoint requires variant='small'."
            )
        expected_frontend = (512, 256, 512)
        observed_frontend = (int(n_fft), int(hop_length), int(win_length))
        if observed_frontend != expected_frontend:
            raise ValueError(
                "Official MetricGAN+ frontend mismatch: expected "
                f"n_fft/hop/win={expected_frontend}, got {observed_frontend}."
            )
        return build_metricgan_standalone(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            variant=variant,
            native8k=False,
            init_from_pretrained=bool(initialize_from_official),
            arch_name=model_family,
            feature_domain="official_log_magnitude",
            window_type="hamming",
            official_checkpoint_sha256=(
                official_checkpoint_sha256
                or METRICGAN_PLUS_CHECKPOINT_SHA256
            ),
            confidence_calibration_enabled=confidence_calibration_enabled,
            confidence_calibration_low=confidence_calibration_low,
            confidence_calibration_high=confidence_calibration_high,
            confidence_calibration_threshold=confidence_calibration_threshold,
            confidence_calibration_temperature=confidence_calibration_temperature,
            adaptive_router_enabled=adaptive_router_enabled,
            adaptive_router_feature_mean=adaptive_router_feature_mean,
            adaptive_router_feature_scale=adaptive_router_feature_scale,
            adaptive_router_weights=adaptive_router_weights,
            adaptive_router_bias=adaptive_router_bias,
            adaptive_router_threshold=adaptive_router_threshold,
            multi_router_enabled=multi_router_enabled,
            multi_router_lows=multi_router_lows,
            multi_router_feature_means=multi_router_feature_means,
            multi_router_feature_scales=multi_router_feature_scales,
            multi_router_weights=multi_router_weights,
            multi_router_biases=multi_router_biases,
            multi_router_threshold=multi_router_threshold,
            multi_router_feature_transform=multi_router_feature_transform,
        )
    if model_family == "metricgan_plus_native8k":
        if spectral_native_gate:
            raise ValueError("MetricGAN+ native8k does not support spectral-native gating.")
        return build_metricgan_standalone(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            variant=variant,
            native8k=True,
            init_from_pretrained=False,
        )
    if model_family in {
        "metricgan_plus_student_wb",
        "metricgan_plus_student_nb",
        "metricgan_plus_student_wb_causal_max",
        "metricgan_plus_student_nb_causal_max",
        "metricgan_plus_native8k_causal_s",
        "metricgan_plus_native8k_causal_xs",
        "metricgan_plus_native8k_causal_n6",
        "metricgan_plus_native8k_causal_max",
    }:
        if spectral_native_gate:
            raise ValueError(f"{model_family} does not support spectral-native gating.")
        if model_family in {
            "metricgan_plus_student_wb",
            "metricgan_plus_student_wb_causal_max",
        }:
            resolve_bandwidth("wb", sample_rate=sample_rate)
        elif model_family in {
            "metricgan_plus_student_nb",
            "metricgan_plus_student_nb_causal_max",
        }:
            resolve_bandwidth("nb", sample_rate=sample_rate)
        return build_metricgan_causal_lite(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            family=model_family,
            qat=qat,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def build_enhancer(
    model_family: str,
    variant: str = "base",
    *,
    spectral_native_gate: bool = False,
    postfilter_mode: str = "none",
    postfilter_preset: str = "medium",
    train_postfilter: bool = False,
    erb_bands: int = 32,
    context_frames: int = 5,
    guidance_classic: str = "none",
    qat: bool = False,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 320,
    initialize_from_official: bool = True,
    official_checkpoint_sha256: str | None = None,
    confidence_calibration_enabled: bool = False,
    confidence_calibration_low: float = 0.0,
    confidence_calibration_high: float = 0.0,
    confidence_calibration_threshold: float = -4.0,
    confidence_calibration_temperature: float = 1.5,
    adaptive_router_enabled: bool = False,
    adaptive_router_feature_mean: list[float] | tuple[float, ...] | None = None,
    adaptive_router_feature_scale: list[float] | tuple[float, ...] | None = None,
    adaptive_router_weights: list[float] | tuple[float, ...] | None = None,
    adaptive_router_bias: float = 0.0,
    adaptive_router_threshold: float = 0.0,
    multi_router_enabled: bool = False,
    multi_router_lows: list[float] | tuple[float, ...] | None = None,
    multi_router_feature_means: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_feature_scales: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_weights: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    multi_router_biases: list[float] | tuple[float, ...] | None = None,
    multi_router_threshold: float = 0.0,
    multi_router_feature_transform: str = "identity",
) -> nn.Module:
    base_model = build_model(
        model_family,
        variant,
        spectral_native_gate=spectral_native_gate,
        erb_bands=erb_bands,
        context_frames=context_frames,
        guidance_classic=guidance_classic,
        qat=qat,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        initialize_from_official=initialize_from_official,
        official_checkpoint_sha256=official_checkpoint_sha256,
        confidence_calibration_enabled=confidence_calibration_enabled,
        confidence_calibration_low=confidence_calibration_low,
        confidence_calibration_high=confidence_calibration_high,
        confidence_calibration_threshold=confidence_calibration_threshold,
        confidence_calibration_temperature=confidence_calibration_temperature,
        adaptive_router_enabled=adaptive_router_enabled,
        adaptive_router_feature_mean=adaptive_router_feature_mean,
        adaptive_router_feature_scale=adaptive_router_feature_scale,
        adaptive_router_weights=adaptive_router_weights,
        adaptive_router_bias=adaptive_router_bias,
        adaptive_router_threshold=adaptive_router_threshold,
        multi_router_enabled=multi_router_enabled,
        multi_router_lows=multi_router_lows,
        multi_router_feature_means=multi_router_feature_means,
        multi_router_feature_scales=multi_router_feature_scales,
        multi_router_weights=multi_router_weights,
        multi_router_biases=multi_router_biases,
        multi_router_threshold=multi_router_threshold,
        multi_router_feature_transform=multi_router_feature_transform,
    )
    postfilter_config = resolve_postfilter_config(postfilter_mode, postfilter_preset)
    if not postfilter_config.enabled:
        return base_model
    return PostFilterEnhancer(
        base_model,
        postfilter_config=postfilter_config,
        apply_in_train=train_postfilter,
    )
