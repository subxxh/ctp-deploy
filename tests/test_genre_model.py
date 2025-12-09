from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genre_model import GenreModel, TrainingConfig, save_artifacts, train_and_evaluate


def _build_dataset(rows: int = 40) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        genre = "classical" if idx % 2 else "blues"
        data.append(
            {
                "filename": f"track_{idx}.wav",
                "chroma_stft_mean": 0.1 * idx,
                "chroma_stft_var": 0.2 * idx + (1 if genre == "classical" else 0),
                "tempo": 90.0 + idx,
                "mfcc1_mean": -150 + idx,
                "genre": genre,
            }
        )
    return pd.DataFrame(data)


def test_training_pipeline_and_inference(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    artifact_dir = tmp_path / "artifacts"
    df = _build_dataset()
    df.to_csv(dataset_path, index=False)

    config = TrainingConfig(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        test_size=0.25,
        random_state=0,
        n_estimators=10,
        max_depth=None,
        n_jobs=1,
    )
    result = train_and_evaluate(config)

    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert "macro_f1" in result.metrics
    save_artifacts(result, artifact_dir)

    model = GenreModel(artifact_dir)
    feature_map = df.drop(columns=["filename", "genre"]).iloc[0].to_dict()

    prediction = model.predict(feature_map)
    assert prediction in df["genre"].unique()

    proba = model.predict_proba(feature_map)
    assert pytest.approx(proba.sum(), rel=1e-6) == 1.0
    assert set(proba.index) == set(df["genre"].unique())

    top2 = model.top_k(feature_map, k=2)
    assert len(top2) == 2
    assert top2[0]["probability"] >= top2[1]["probability"]

    with pytest.raises(KeyError):
        model.predict({})
