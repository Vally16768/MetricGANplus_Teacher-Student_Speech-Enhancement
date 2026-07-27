"""Validation for the canonical VoiceBank+DEMAND WB/NB campaign."""

from __future__ import annotations

from typing import Any

from sebench.bandwidth import validate_frontend


VOICEBANK_DATASET_ID = "VoiceBank+DEMAND"


def _validate_model_profile(
    name: str,
    payload: dict[str, Any],
    *,
    expected_family: str,
) -> dict[str, Any]:
    if payload.get("family") != expected_family:
        raise ValueError(
            f"{name}: canonical family must be {expected_family}, got "
            f"{payload.get('family') or 'missing'}."
        )
    profile = validate_frontend(
        str(payload["bandwidth"]),
        sample_rate=int(payload["sample_rate"]),
        n_fft=int(payload["n_fft"]),
        hop_length=int(payload["hop_length"]),
        win_length=int(payload["win_length"]),
    )
    reference = str(payload.get("evaluation_reference_bandwidth") or "").lower()
    if reference != profile.name:
        raise ValueError(
            f"{name}: evaluation reference must be {profile.name.upper()}, got "
            f"{reference or 'missing'}."
        )
    return profile.as_dict()


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
        expected_family="metricgan_plus_teacher_wb",
    )
    if teacher["name"] != "wb":
        raise ValueError("The canonical teacher must be WB/16000 Hz.")

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
    proxy_profiles = set(discriminator.get("separate_proxy_profiles") or [])
    if proxy_profiles != {"wb", "nb"}:
        raise ValueError(
            "WB and NB must use separate, bandwidth-matched metric proxies."
        )

    ablations = list(plan.get("ablations") or [])
    required = {
        "teacher_baseline",
        "teacher_metric",
        "student_wb_baseline",
        "student_wb_metric",
        "student_nb_baseline",
        "student_nb_metric",
    }
    names = {str(item.get("name")) for item in ablations if isinstance(item, dict)}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing canonical ablations: {', '.join(missing)}")

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
            "separate_proxy_profiles": ["nb", "wb"],
        },
        "ablation_count": len(names),
        "tts_status": tts["status"],
        "valid": True,
    }
