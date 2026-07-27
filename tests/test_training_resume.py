from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.training import (  # noqa: E402
    _capture_rng_state,
    _metric_discriminator_state_fields,
    _resume_epoch_position,
    _restore_rng_state,
    _restore_metric_discriminator_state,
    _training_control_state_fields,
)
from sebench.losses import SpeechBrainMetricDiscriminator  # noqa: E402


class ResumeControlStateTests(unittest.TestCase):
    @staticmethod
    def _run_scores(
        scores: list[float],
        *,
        state_path: Path | None = None,
        interrupt_after: int | None = None,
    ) -> dict[str, object]:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
        best_score = float("-inf")
        best_epoch = 0
        epochs_without_improve = 0
        history_rows: list[dict[str, object]] = []
        best_model_state: dict[str, torch.Tensor] = {}
        start_epoch = 1

        if state_path is not None and state_path.is_file():
            payload = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            best_score = float(payload["best_score"])
            best_epoch = int(payload["best_epoch"])
            epochs_without_improve = int(payload["epochs_without_improve"])
            history_rows = list(payload["history_rows"])
            best_model_state = copy.deepcopy(payload["best_model_state"])
            start_epoch, _ = _resume_epoch_position(payload)

        for epoch in range(start_epoch, len(scores) + 1):
            score = float(scores[epoch - 1])
            with torch.no_grad():
                model.weight.add_(float(epoch))
            scheduler.step(score)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                epochs_without_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improve += 1
            history_rows.append(
                {
                    "epoch": epoch,
                    "selection_score": score,
                    "epochs_without_improve": epochs_without_improve,
                    "lr_after_eval": optimizer.param_groups[0]["lr"],
                }
            )
            control = _training_control_state_fields(
                scheduler=scheduler,
                best_score=best_score,
                best_epoch=best_epoch,
                best_rank_metrics={},
                best_select_metrics={"pesq_mean": best_score},
                best_rank_metrics_by_split={},
                best_select_metrics_by_split={},
                epochs_without_improve=epochs_without_improve,
                history_rows=history_rows,
            )
            payload = {
                "epoch": epoch,
                "reason": "evaluation",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_model_state": best_model_state,
                **control,
            }
            if interrupt_after == epoch:
                if state_path is None:
                    raise AssertionError("state_path is required for interruption")
                torch.save(payload, state_path)
                return {"interrupted": True}

        return {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epochs_without_improve": epochs_without_improve,
            "history_rows": history_rows,
            "best_model_state": best_model_state,
            "scheduler_state": scheduler.state_dict(),
        }

    def test_interrupt_resume_matches_uninterrupted_post_evaluation_state(
        self,
    ) -> None:
        scores = [1.0, 0.9, 0.8, 1.1, 1.0]
        uninterrupted = self._run_scores(scores)
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "training_state.pt"
            interrupted = self._run_scores(
                scores,
                state_path=state_path,
                interrupt_after=2,
            )
            resumed = self._run_scores(scores, state_path=state_path)

        self.assertTrue(interrupted["interrupted"])
        self.assertEqual(resumed["lr"], uninterrupted["lr"])
        self.assertEqual(
            resumed["epochs_without_improve"],
            uninterrupted["epochs_without_improve"],
        )
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        self.assertEqual(resumed["best_score"], uninterrupted["best_score"])
        self.assertEqual(
            resumed["scheduler_state"],
            uninterrupted["scheduler_state"],
        )
        self.assertEqual(
            resumed["history_rows"],
            uninterrupted["history_rows"],
        )
        for key, value in uninterrupted["best_model_state"].items():
            self.assertTrue(
                torch.equal(resumed["best_model_state"][key], value),
                key,
            )

    def test_evaluation_state_resumes_at_next_epoch(self) -> None:
        self.assertEqual(
            _resume_epoch_position({"epoch": 7, "reason": "evaluation"}),
            (8, 7),
        )

    def test_rng_state_is_weights_only_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "rng.pt"
            torch.save({"rng_state": _capture_rng_state()}, path)
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        _restore_rng_state(payload["rng_state"])

    def test_metric_discriminator_optimizer_history_and_replay_resume(
        self,
    ) -> None:
        source = SpeechBrainMetricDiscriminator(base_channels=2)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=2e-4)
        waveform = torch.randn(1, 8192)
        loss = source.normalized_score(waveform, waveform).mean()
        loss.backward()
        source_optimizer.step()
        history = [
            {
                "epoch": 1,
                "calibration_gate": {"passed": True},
                "replay_index": "epoch_0001/index.json",
            }
        ]
        with tempfile.TemporaryDirectory() as raw:
            replay = Path(raw) / "replay"
            replay.mkdir()
            payload = _metric_discriminator_state_fields(
                discriminator=source,
                optimizer=source_optimizer,
                refresh_history=history,
                replay_root=replay.as_posix(),
            )
            state_path = Path(raw) / "state.pt"
            torch.save(payload, state_path)
            loaded = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
            target = SpeechBrainMetricDiscriminator(base_channels=2)
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
            restored_history = _restore_metric_discriminator_state(
                loaded,
                discriminator=target,
                optimizer=target_optimizer,
                replay_root=replay.as_posix(),
            )
            for key, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
            source_optim_state = source_optimizer.state_dict()
            target_optim_state = target_optimizer.state_dict()
            self.assertEqual(
                source_optim_state["param_groups"],
                target_optim_state["param_groups"],
            )
            for parameter_id, state in source_optim_state["state"].items():
                for key, value in state.items():
                    observed = target_optim_state["state"][parameter_id][key]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, observed), key)
                    else:
                        self.assertEqual(value, observed)
            self.assertEqual(restored_history, history)
            with self.assertRaisesRegex(ValueError, "replay identity"):
                _restore_metric_discriminator_state(
                    loaded,
                    discriminator=target,
                    optimizer=target_optimizer,
                    replay_root=(Path(raw) / "other").as_posix(),
                )


if __name__ == "__main__":
    unittest.main()
