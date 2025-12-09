"""CLI entrypoint for training the GTZAN genre classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# from genre_model import TrainingConfig, save_artifacts, train_and_evaluate
from genre_model.config import TrainingConfig
from genre_model.training import save_artifacts, train_and_evaluate


_DEFAULT_DATASET = Path("gtzan_training/features_30_cleaned.csv")
_DEFAULT_ARTIFACT_DIR = Path("genre_model/artifacts")


def _optional_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def _optional_str(value: str) -> str | None:
    if value.lower() == "none":
        return None
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GTZAN genre classifier and persist artifacts.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET, help="Path to the cleaned GTZAN feature CSV")
    parser.add_argument("--artifacts", type=Path, default=_DEFAULT_ARTIFACT_DIR, help="Directory to store trained artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=500, help="RandomForest n_estimators")
    parser.add_argument("--max-depth", type=_optional_int, default=None, help="Max tree depth (None = unrestricted)")
    parser.add_argument("--min-samples-split", type=int, default=2, help="RandomForest min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="RandomForest min_samples_leaf")
    parser.add_argument("--max-features", type=str, default="sqrt", help="RandomForest max_features setting")
    parser.add_argument("--class-weight", type=_optional_str, default="balanced", help="RandomForest class_weight setting")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallelism for training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = TrainingConfig(
        dataset_path=args.dataset,
        artifact_dir=args.artifacts,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
    )
    result = train_and_evaluate(config)
    save_artifacts(result, config.artifact_dir)
    acc = result.metrics.get("accuracy", 0.0)
    print(f"Validation accuracy: {acc:.3f}")
    print(f"Artifacts saved to {config.artifact_dir}")


if __name__ == "__main__":
    main()
