"""Validation for the canonical VoiceBank+DEMAND WB/NB campaign."""

from __future__ import annotations

from typing import Any

from sebench.bandwidth import resolve_bandwidth, validate_frontend


VOICEBANK_DATASET_ID = "VoiceBank+DEMAND"


def _validate_model_profile(
    name: str,
    payload: dict[str, Any],
    *,
    expected_family: str,
    canonical_frontend: bool = True,
) -> dict[str, Any]:
    if payload.get("family") != expected_family:
        raise ValueError(
            f"{name}: canonical family must be {expected_family}, got "
            f"{payload.get('family') or 'missing'}."
        )
    if canonical_frontend:
        profile = validate_frontend(
            str(payload["bandwidth"]),
            sample_rate=int(payload["sample_rate"]),
            n_fft=int(payload["n_fft"]),
            hop_length=int(payload["hop_length"]),
            win_length=int(payload["win_length"]),
        )
    else:
        profile = resolve_bandwidth(
            str(payload["bandwidth"]),
            sample_rate=int(payload["sample_rate"]),
        )
    reference = str(payload.get("evaluation_reference_bandwidth") or "").lower()
    if reference != profile.name:
        raise ValueError(
            f"{name}: evaluation reference must be {profile.name.upper()}, got "
            f"{reference or 'missing'}."
        )
    result = profile.as_dict()
    result.update(
        {
            "n_fft": int(payload["n_fft"]),
            "hop_length": int(payload["hop_length"]),
            "win_length": int(payload["win_length"]),
        }
    )
    return result


def validate_research_plan(plan: dict[str, Any]) -> dict[str, Any]:
    dataset = dict(plan.get("dataset") or {})
    if dataset.get("name") != VOICEBANK_DATASET_ID:
        raise ValueError(
            f"Canonical training dataset must be exactly {VOICEBANK_DATASET_ID}."
        )
    if not bool(dataset.get("read_only", False)):
        raise ValueError("The VoiceBank+DEMAND source dataset must be read-only.")

    teacher = _validate_model_profile(
        "teacher",
        dict(plan["teacher"]),
        expected_family="metricgan_plus_teacher_official_wb",
        canonical_frontend=False,
    )
    if teacher["name"] != "wb":
        raise ValueError("The canonical teacher must be WB/16000 Hz.")
    teacher_payload = dict(plan["teacher"])
    observed_teacher_frontend = (
        int(teacher_payload["n_fft"]),
        int(teacher_payload["hop_length"]),
        int(teacher_payload["win_length"]),
    )
    if observed_teacher_frontend != (512, 256, 512):
        raise ValueError(
            "The official MetricGAN+ teacher requires frontend 512/256/512."
        )

    students_payload = dict(plan.get("students") or {})
    if set(students_payload) != {"wb", "nb"}:
        raise ValueError("The canonical plan requires exactly WB and NB students.")
    students = {
        name: _validate_model_profile(
            f"student.{name}",
            dict(payload),
            expected_family=f"metricgan_plus_student_{name}_causal_max",
        )
        for name, payload in students_payload.items()
    }
    if students["wb"]["name"] != "wb" or students["nb"]["name"] != "nb":
        raise ValueError("Student keys and bandwidth profiles must match.")

    discriminator = dict(plan.get("metric_discriminator") or {})
    if discriminator.get("target_metric") != "PESQ":
        raise ValueError("The first canonical metric discriminator target is PESQ.")
    proxy_profiles = list(discriminator.get("proxy_profiles") or [])
    if proxy_profiles != ["wb"]:
        raise ValueError(
            "The canonical teacher-fine-tuning campaign requires one WB proxy."
        )
    if discriminator.get("scope") != "teacher_finetune_only":
        raise ValueError("The canonical PESQ proxy is scoped to teacher fine-tuning.")

    stages = list(plan.get("stages") or [])
    required = {
        "official_teacher_baseline",
        "student_wb_from_official",
        "student_nb_from_official",
        "teacher_finetune_control",
        "teacher_finetune_metric",
        "student_wb_from_improved",
        "student_nb_from_improved",
    }
    names = {str(item.get("name")) for item in stages if isinstance(item, dict)}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing canonical stages: {', '.join(missing)}")

    cache = dict(plan.get("teacher_cache") or {})
    if cache.get("location") != "desktop_local_only":
        raise ValueError("Teacher caches must be Desktop-local.")
    if bool(cache.get("cache_inputs", True)):
        raise ValueError("Teacher caches must not duplicate noisy/clean inputs.")
    if cache.get("storage_dtype") != "float16":
        raise ValueError("Canonical teacher-cache storage must be float16.")

    tts = dict(plan.get("tts_extension") or {})
    if tts.get("status") != "planned_separate_domain_calibration":
        raise ValueError(
            "TTS transfer must remain planned until its metric proxy is "
            "recalibrated and validated on TTS generator outputs."
        )

    return {
        "dataset": VOICEBANK_DATASET_ID,
        "teacher": teacher,
        "students": students,
        "metric_discriminator": {
            "target_metric": "PESQ",
            "proxy_profiles": ["wb"],
            "scope": "teacher_finetune_only",
        },
        "stage_count": len(names),
        "tts_status": tts["status"],
        "valid": True,
    }
