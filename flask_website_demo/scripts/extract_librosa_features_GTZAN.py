"""CLI helper to extract GTZAN-compatible audio features using librosa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict

import librosa
import numpy as np
import pandas as pd


def _matrix_stats(matrix: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute mean and variance for each row of a feature matrix."""
    matrix = np.atleast_2d(matrix)
    stats: Dict[str, float] = {}
    for idx, row in enumerate(matrix, start=1):
        stats[f"{prefix}_mean_{idx}"] = float(np.mean(row))
        stats[f"{prefix}_var_{idx}"] = float(np.var(row))
    return stats


def extract_features(
    audio_path: Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> Dict[str, float]:
    y, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=duration,
        res_type="kaiser_fast",
    )
    if not np.any(y):
        raise ValueError(f"{audio_path} is silent or empty")

    feats: Dict[str, float] = {}

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    feats["chroma_stft_mean"] = np.mean(chroma_stft).item()
    feats["chroma_stft_var"] = np.var(chroma_stft).item()

    # Spectral features
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)
    feats["loudness"] = np.mean(rmse).item()
    feats["loudness_variance"] = np.var(rmse).item()

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    feats["spectral_centroid_mean"] = np.mean(spectral_centroid).item()
    feats["spectral_centroid_var"] = np.var(spectral_centroid).item()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    feats["spectral_bandwidth_var"] = np.var(spectral_bandwidth).item()

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    feats["rolloff_var"] = np.var(spectral_rolloff).item()

    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    feats["zero_crossing_rate_mean"] = np.mean(zcr).item()
    feats["zero_crossing_rate_var"] = np.var(zcr).item()

    # Harmony
    y_harmonic = librosa.effects.harmonic(y)
    harmony = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    feats["harmony_mean"] = np.mean(harmony).item()
    feats["harmony_var"] = np.var(harmony).item()

    # Perceptual features: spectral contrast as proxy for "perceptr"
    perceptr = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    feats["perceptr_mean"] = np.mean(perceptr).item()
    feats["perceptr_var"] = np.var(perceptr).item()

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = tempo.item()
    
    # MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    for idx, row in enumerate(mfcc, start=1):
        feats[f"mfcc{idx}_mean"] = np.mean(row).item()
        feats[f"mfcc{idx}_var"] = np.var(row).item()
    
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
    sample_rate: int = 22050,
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
        description="Extract GTZAN-compatible audio features using librosa."
    )
    parser.add_argument(
        "inputs", nargs="+", help="Audio files or directories to process"
    )
    parser.add_argument(
        "--pattern", default="**/*.wav", help="Glob pattern for directories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gtzan_features.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Target sampling rate"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for frame-based features",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optionally crop each file to first N seconds",
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
