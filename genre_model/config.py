"""Configuration and constants for the GTZAN genre model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TARGET_COLUMN = "genre"
DROP_COLUMNS: Sequence[str] = ("filename",)
MODEL_FILENAME = "genre_classifier.joblib"
FEATURE_NAMES_FILENAME = "feature_names.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    artifact_dir: Path = Path("genre_model/artifacts")
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    class_weight: str | None = "balanced"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
