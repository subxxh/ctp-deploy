"""Training utilities for the GTZAN genre classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DROP_COLUMNS,
    FEATURE_NAMES_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    TARGET_COLUMN,
    TrainingConfig,
)


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, float]
    report: str


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df:
        raise ValueError(f"Dataset at {path} is missing '{TARGET_COLUMN}' column")
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    columns = [col for col in df.columns if col not in DROP_COLUMNS and col != TARGET_COLUMN]
    if not columns:
        raise ValueError("Dataset does not contain feature columns")
    return columns


def _build_pipeline(config: TrainingConfig) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def train_and_evaluate(config: TrainingConfig) -> TrainingResult:
    df = load_dataset(config.dataset_path)
    feature_names = _feature_columns(df)

    X = df[feature_names]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = _build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }
    report_text = classification_report(y_val, y_pred, zero_division=0)

    return TrainingResult(pipeline=pipeline, feature_names=feature_names, metrics=metrics, report=report_text)


def save_artifacts(result: TrainingResult, artifact_dir: Path | str) -> None:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.pipeline, artifact_path / MODEL_FILENAME)

    with (artifact_path / FEATURE_NAMES_FILENAME).open("w", encoding="utf-8") as fp:
        json.dump(result.feature_names, fp)

    metrics_payload = result.metrics | {"classification_report": result.report}
    with (artifact_path / METRICS_FILENAME).open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)


def load_feature_names(feature_file: Path) -> list[str]:
    with feature_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Feature names file at {feature_file} is not a list")
    return [str(item) for item in data]
