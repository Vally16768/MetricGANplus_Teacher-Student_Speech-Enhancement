"""Predeclared eight-action quadratic router for T16."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sebench.t14_quadratic_router import (
    fit_quadratic_metric_ridges,
    run_t14_quadratic_search,
)


T16_ACTION_LOWS = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8)


def _fit_t16_ridges(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return fit_quadratic_metric_ridges(records, lows=T16_ACTION_LOWS)


def run_t16_fine_action_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t10_summary_path: str | Path,
    t11_summary_path: str | Path,
    t14_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    return run_t14_quadratic_search(
        teacher_checkpoint=teacher_checkpoint,
        t9_checkpoint=t9_checkpoint,
        t9_summary_path=t9_summary_path,
        t10_summary_path=t10_summary_path,
        t11_summary_path=t11_summary_path,
        t13_summary_path=t14_summary_path,
        val_rank_manifest=val_rank_manifest,
        val_select_manifest=val_select_manifest,
        baseline_rank_metrics=baseline_rank_metrics,
        baseline_select_metrics=baseline_select_metrics,
        output_dir=output_dir,
        device=device,
        max_eval_files=max_eval_files,
        progress_callback=progress_callback,
        metric_ridge_fitter=_fit_t16_ridges,
        strategy="T16-FINE-ACTION-QUADRATIC-MULTIOBJECTIVE",
        checkpoint_filename="T16-FINE-ACTION-ROUTED.pt",
        prerequisite_name="T15",
        action_lows=T16_ACTION_LOWS,
    )
