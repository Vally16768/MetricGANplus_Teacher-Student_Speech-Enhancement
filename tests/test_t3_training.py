from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code_and_documentation"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sebench.t3_training import (  # noqa: E402
    atomic_torch_save,
    build_t3_training_state,
    clone_state_dict,
    restore_t3_training_state,
)


class T3TrainingStateTests(unittest.TestCase):
    def assert_nested_equal(self, left: object, right: object) -> None:
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertIsInstance(right, dict)
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertIsInstance(right, type(left))
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def _components() -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=0,
            min_lr=1e-5,
        )
        return model, optimizer, scheduler

    @staticmethod
    def _step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        score: float,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(3, 4)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step(score)

    def test_post_evaluation_resume_is_bit_exact(self) -> None:
        provenance = {"checkpoint": "abc", "seed": 3003}
        torch.manual_seed(17)
        control_model, control_optimizer, control_scheduler = self._components()
        initial = clone_state_dict(control_model)
        self._step(control_model, control_optimizer, control_scheduler, 2.0)
        first_state = build_t3_training_state(
            model=control_model,
            optimizer=control_optimizer,
            scheduler=control_scheduler,
            branch="E2-PMSQE",
            proposal=1,
            accepted_epoch=1,
            best_epoch=1,
            best_score=2.0,
            best_model_state=clone_state_dict(control_model),
            epochs_without_improve=0,
            consecutive_rejections=0,
            history=[{"proposal": 1, "accepted": True}],
            provenance=provenance,
        )
        self._step(control_model, control_optimizer, control_scheduler, 1.9)
        control_final = clone_state_dict(control_model)

        resumed_model, resumed_optimizer, resumed_scheduler = self._components()
        resumed_model.load_state_dict(initial)
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "state.pt"
            atomic_torch_save(first_state, state_path)
            loaded = torch.load(state_path, map_location="cpu", weights_only=False)
        restored = restore_t3_training_state(
            loaded,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            expected_branch="E2-PMSQE",
            expected_provenance=provenance,
        )
        self.assertEqual(restored["accepted_epoch"], 1)
        self._step(resumed_model, resumed_optimizer, resumed_scheduler, 1.9)
        for key, expected in control_final.items():
            self.assertTrue(torch.equal(resumed_model.state_dict()[key], expected), key)
        self.assert_nested_equal(
            resumed_optimizer.state_dict(),
            control_optimizer.state_dict(),
        )
        self.assert_nested_equal(
            resumed_scheduler.state_dict(),
            control_scheduler.state_dict(),
        )

    def test_resume_rejects_provenance_mismatch(self) -> None:
        model, optimizer, scheduler = self._components()
        payload = build_t3_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            branch="E1-SUP",
            proposal=0,
            accepted_epoch=0,
            best_epoch=0,
            best_score=2.0,
            best_model_state=clone_state_dict(model),
            epochs_without_improve=0,
            consecutive_rejections=0,
            history=[],
            provenance={"checkpoint": "abc"},
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            restore_t3_training_state(
                payload,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_branch="E1-SUP",
                expected_provenance={"checkpoint": "changed"},
            )


if __name__ == "__main__":
    unittest.main()
