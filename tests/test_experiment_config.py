from __future__ import annotations

import json
from pathlib import Path

import torch

from materials_gnn.utils import write_experiment_config


def test_write_experiment_config_handles_paths_and_tensors(tmp_path: Path) -> None:
    path = write_experiment_config(
        {
            "path": tmp_path / "data.csv",
            "scalar": torch.tensor(1.5),
            "vector": torch.tensor([1, 2, 3]),
        },
        tmp_path / "experiment_config.json",
    )

    payload = json.loads(path.read_text())
    assert payload["path"].endswith("data.csv")
    assert payload["scalar"] == 1.5
    assert payload["vector"] == [1, 2, 3]
