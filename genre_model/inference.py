"""Inference utilities for serving GTZAN genre predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

from .config import FEATURE_NAMES_FILENAME, MODEL_FILENAME
from .training import load_feature_names


class GenreModel:
    """Loadable inference helper intended for Flask routes."""

    def __init__(self, artifact_dir: Path | str = Path("genre_model/artifacts")) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.pipeline = joblib.load(self.artifact_dir / MODEL_FILENAME)
        self.feature_names = load_feature_names(self.artifact_dir / FEATURE_NAMES_FILENAME)
        self.classes_ = self.pipeline.named_steps["clf"].classes_.tolist()

    def _feature_frame(self, feature_map: Mapping[str, float]) -> pd.DataFrame:
        row: dict[str, float] = {}
        for name in self.feature_names:
            if name not in feature_map:
                raise KeyError(f"Missing feature '{name}' for inference")
            row[name] = float(feature_map[name])
        return pd.DataFrame([row])

    def predict(self, feature_map: Mapping[str, float]) -> str:
        frame = self._feature_frame(feature_map)
        return str(self.pipeline.predict(frame)[0])

    def predict_proba(self, feature_map: Mapping[str, float]) -> pd.Series:
        frame = self._feature_frame(feature_map)
        proba = self.pipeline.predict_proba(frame)[0]
        return pd.Series(proba, index=self.classes_)

    def top_k(self, feature_map: Mapping[str, float], *, k: int = 3) -> list[dict[str, float]]:
        probs = self.predict_proba(feature_map).sort_values(ascending=False).head(k)
        return [{"genre": label, "probability": float(prob)} for label, prob in probs.items()]
