from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch


@dataclass
class ApplicationState:
    global_step: int
    data_cursor: int
    dataset_fingerprint: str
    config_fingerprint: str
    scheduler_state: dict[str, Any]
    python_rng: object
    numpy_rng: object
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    consumed_sample_ids: list[int] = field(default_factory=list)

    @classmethod
    def capture(
        cls,
        step: int,
        cursor: int,
        dataset: str,
        config: str,
        scheduler: Any,
        consumed_sample_ids: list[int] | None = None,
    ) -> ApplicationState:
        return cls(
            step,
            cursor,
            dataset,
            config,
            scheduler.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            list(consumed_sample_ids or []),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def restore_runtime(self, scheduler: Any) -> None:
        scheduler.load_state_dict(self.scheduler_state)
        random.setstate(cast(tuple[Any, ...], self.python_rng))
        np.random.set_state(cast(Any, self.numpy_rng))
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)


def dcp_save(path: str, model: Any, optimizer: Any, app: ApplicationState) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state, "application": app}, checkpoint_id=path
    )


def dcp_load(
    path: str,
    model: Any,
    optimizer: Any,
    app: ApplicationState,
    scheduler: Any | None = None,
    *,
    expected_dataset_fingerprint: str | None = None,
    expected_config_fingerprint: str | None = None,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state, "application": app}
    dcp.load(state, checkpoint_id=path)
    if (
        expected_dataset_fingerprint is not None
        and app.dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise ValueError("checkpoint dataset fingerprint mismatch")
    if (
        expected_config_fingerprint is not None
        and app.config_fingerprint != expected_config_fingerprint
    ):
        raise ValueError("checkpoint config fingerprint mismatch")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=cast(Any, state["model"]),
        optim_state_dict=cast(Any, state["optimizer"]),
    )
    if scheduler is not None:
        app.restore_runtime(scheduler)
