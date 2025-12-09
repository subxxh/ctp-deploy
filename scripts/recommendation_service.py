"""Glue code between librosa feature extraction and the Annoy similarity engine."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from annoy_similarity.annoy_engine import SpotifyAnnoyEngine
from scripts.extract_librosa_features import extract_features


_FEATURE_LAYOUT: Sequence[tuple[str, int]] = (
    ("mfcc", 20),
    ("chroma_cqt", 12),
    ("spectral_centroid", 1),
    ("spectral_bandwidth", 1),
    ("spectral_rolloff", 1),
    ("rmse", 1),
    ("zcr", 1),
)


def _expected_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for prefix, count in _FEATURE_LAYOUT:
        names.extend(f"{prefix}_mean_{i}" for i in range(1, count + 1))
        names.extend(f"{prefix}_std_{i}" for i in range(1, count + 1))
    return tuple(names)


_FEATURE_NAMES = _expected_feature_names()


def _vector_from_feature_map(feature_map: Dict[str, float]) -> list[float]:
    try:
        return [feature_map[name] for name in _FEATURE_NAMES]
    except KeyError as exc:  # pragma: no cover - defensive guardrail
        missing = exc.args[0]
        raise KeyError(f"Missing expected feature '{missing}'") from None


class RecommendationService:
    """Provides similarity lookups from raw audio or track metadata."""

    def __init__(
        self,
        model_dir: str = "annoy_similarity/model/",
        *,
        engine: SpotifyAnnoyEngine | None = None,
    ) -> None:
        self.engine = engine or SpotifyAnnoyEngine(model_dir=model_dir)

    def recommend_from_audio(
        self,
        audio_path: Path | str,
        *,
        k: int = 10,
        sample_rate: int = 22_050,
        hop_length: int = 512,
        duration: float | None = None,
    ) -> pd.DataFrame:
        path = Path(audio_path)
        feature_map = extract_features(
            path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            duration=duration,
        )
        return self.recommend_from_features(feature_map, k=k)

    def recommend_from_features(
        self,
        feature_map: Dict[str, float],
        *,
        k: int = 10,
    ) -> pd.DataFrame:
        vector = _vector_from_feature_map(feature_map)
        meta, distances = self.engine.query_by_features(vector, k=k)
        result = meta.copy()
        result["distance"] = distances
        return result

    def recommend_from_track_id(self, track_id: int | str, *, k: int = 10) -> pd.DataFrame:
        idx = self.engine.lookup(track_id)
        meta, distances = self.engine.query_by_track_index(idx, k=k)
        result = meta.copy()
        result["distance"] = distances
        return result

    def build_response_payload(
        self,
        rows: pd.DataFrame,
        *,
        fields: Iterable[str] | None = None,
    ) -> list[dict]:
        columns = list(fields) if fields is not None else rows.columns.tolist()
        return rows[columns].to_dict("records")
