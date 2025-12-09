"""CLI helper to reproduce FMA audio features with librosa."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import librosa
import numpy as np
import pandas as pd


def _matrix_stats(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    matrix = np.atleast_2d(matrix)
    stats: dict[str, float] = {}
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    for idx, value in enumerate(means, start=1):
        stats[f"{prefix}_mean_{idx}"] = float(value)
    for idx, value in enumerate(stds, start=1):
        stats[f"{prefix}_std_{idx}"] = float(value)
    return stats


def extract_features(
    audio_path: Path,
    *,
    sample_rate: int = 22_050,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> dict[str, float]:
    y, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=duration,
        res_type="kaiser_fast",
    )
    if not np.any(y):
        raise ValueError(f"{audio_path} is silent or empty")

    feats: dict[str, float] = {}
    feats.update(_matrix_stats(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length), "mfcc"))
    feats.update(_matrix_stats(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length), "chroma_cqt"))
    feats.update(_matrix_stats(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length), "spectral_centroid"))
    feats.update(_matrix_stats(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length), "spectral_bandwidth"))
    feats.update(_matrix_stats(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length), "spectral_rolloff"))
    feats.update(_matrix_stats(librosa.feature.rms(y=y, hop_length=hop_length), "rmse"))
    feats.update(_matrix_stats(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length), "zcr"))
    return feats


def _iter_audio_paths(inputs: Iterable[str], pattern: str) -> Iterator[Path]:
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.rglob(pattern))
        elif path.is_file():
            yield path
        else:
            print(f"[WARN] Skipping missing path: {path}", file=sys.stderr)


def extract_many(
    inputs: Iterable[str],
    *,
    pattern: str = "**/*.wav",
    sample_rate: int = 22_050,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    for audio_path in _iter_audio_paths(inputs, pattern):
        try:
            feats = extract_features(
                audio_path,
                sample_rate=sample_rate,
                hop_length=hop_length,
                duration=duration,
            )
        except Exception as exc:
            print(f"[WARN] Failed on {audio_path}: {exc}", file=sys.stderr)
            continue
        rows.append({"source_path": str(audio_path)} | feats)
    if not rows:
        raise RuntimeError("No audio files produced features")
    return pd.DataFrame(rows)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract librosa features matching fma_metadata_features_joined.csv (without track_id)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio files or directories to process",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.wav",
        help="Glob used when walking directories (default: **/*.wav)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/librosa_features.csv"),
        help="Where to write the feature table",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22_050,
        help="Target sampling rate",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for frame based features",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optionally crop each file to the first N seconds",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    df = extract_many(
        args.inputs,
        pattern=args.pattern,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        duration=args.duration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
