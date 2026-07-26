from __future__ import annotations

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
    "metricgan_plus_student_wb",
    "metricgan_plus_student_nb",
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
    "metricgan_plus_student_wb": 12,
    "metricgan_plus_student_nb": 12,
    "metricgan_plus_native8k": 8,
    "metricgan_plus_native8k_causal_s": 12,
    "metricgan_plus_native8k_causal_xs": 14,
    "metricgan_plus_native8k_causal_n6": 10,
    "metricgan_plus_native8k_causal_max": 8,
}
METRICGAN_PLUS_SOURCE = "speechbrain/metricgan-plus-voicebank"
METRICGAN_PLUS_CACHE_DIR = Path.home() / ".cache" / "sebench" / "metricgan_plus_voicebank"


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.blstm(features)
        encoded = self.activation(self.linear1(encoded))
        encoded = self.linear2(encoded)
        return self.learnable_sigmoid(encoded)


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
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
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
        }
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        original_length = input.shape[-1]
        spec = self._stft(input)
        magnitude = spec.abs().clamp_min(1e-8).pow(0.5)
        features = magnitude.transpose(1, 2)
        mask = self.mask_generator(features).transpose(1, 2).clamp_min(0.0)
        enhanced_magnitude = (mask * magnitude).pow(2.0)
        enhanced_spec = torch.polar(enhanced_magnitude, torch.angle(spec))
        enhanced = self._istft(enhanced_spec, original_length)
        return enhanced[..., :original_length]


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
        "metricgan_plus_native8k_causal_s",
        "metricgan_plus_native8k_causal_xs",
        "metricgan_plus_native8k_causal_n6",
        "metricgan_plus_native8k_causal_max",
    }:
        if spectral_native_gate:
            raise ValueError(f"{model_family} does not support spectral-native gating.")
        if model_family == "metricgan_plus_student_wb":
            resolve_bandwidth("wb", sample_rate=sample_rate)
        elif model_family == "metricgan_plus_student_nb":
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
    )
    postfilter_config = resolve_postfilter_config(postfilter_mode, postfilter_preset)
    if not postfilter_config.enabled:
        return base_model
    return PostFilterEnhancer(
        base_model,
        postfilter_config=postfilter_config,
        apply_in_train=train_postfilter,
    )
