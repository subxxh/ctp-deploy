import pandas as pd
import pytest

from scripts.recommendation_service import (
    RecommendationService,
    _FEATURE_NAMES,
    _vector_from_feature_map,
)


def test_vector_from_feature_map_matches_expected_order():
    feature_map = {name: idx for idx, name in enumerate(_FEATURE_NAMES, start=1)}
    vector = _vector_from_feature_map(feature_map)
    assert vector == list(range(1, len(_FEATURE_NAMES) + 1))


class DummyEngine:
    def __init__(self):
        self.last_vector = None

    def query_by_features(self, vector, k=10):
        self.last_vector = vector
        meta = pd.DataFrame(
            [{"track_id": i, "track_title": f"Song {i}"} for i in range(k)]
        )
        distances = [i / 10 for i in range(k)]
        return meta, distances

    def query_by_track_index(self, idx, k=10):
        meta = pd.DataFrame(
            [{"track_id": idx + i, "track_title": f"Song {idx + i}"} for i in range(k)]
        )
        distances = [i / 5 for i in range(k)]
        return meta, distances

    def lookup(self, track_id):
        return int(track_id)


@pytest.fixture
def dummy_features():
    return {name: float(idx) for idx, name in enumerate(_FEATURE_NAMES, start=1)}


def test_recommend_from_features_uses_engine(dummy_features):
    engine = DummyEngine()
    service = RecommendationService(engine=engine)

    result = service.recommend_from_features(dummy_features, k=3)

    assert engine.last_vector == _vector_from_feature_map(dummy_features)
    assert list(result["distance"]) == [0.0, 0.1, 0.2]


def test_recommend_from_track_id_relays_engine_calls():
    engine = DummyEngine()
    service = RecommendationService(engine=engine)

    result = service.recommend_from_track_id(track_id=5, k=2)

    assert list(result["track_id"]) == [5, 6]
    assert list(result["distance"]) == [0.0, 0.2]
