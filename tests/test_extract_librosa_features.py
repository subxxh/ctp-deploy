import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts.extract_librosa_features import (
    _matrix_stats,
    extract_features,
    extract_many,
)


@pytest.fixture
def tone_writer(tmp_path):
    def _write(filename: str = "tone.wav", seconds: float = 0.5, freq: float = 440.0) -> Path:
        sr = 22_050
        samples = int(sr * seconds)
        t = np.linspace(0, seconds, samples, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * freq * t)
        path = tmp_path / filename
        sf.write(path, y, sr)
        return path

    return _write


def test_matrix_stats_produces_mean_and_std_keys():
    matrix = np.array([[1.0, 3.0], [2.0, 4.0]])
    stats = _matrix_stats(matrix, "foo")
    assert stats["foo_mean_1"] == pytest.approx(2.0)
    assert stats["foo_mean_2"] == pytest.approx(3.0)
    assert stats["foo_std_1"] == pytest.approx(1.0)
    assert stats["foo_std_2"] == pytest.approx(1.0)


def test_extract_features_includes_all_expected_prefixes(tone_writer):
    audio_path = tone_writer()
    feats = extract_features(audio_path)
    required_keys = {
        *{f"mfcc_mean_{i}" for i in range(1, 21)},
        *{f"mfcc_std_{i}" for i in range(1, 21)},
        "chroma_cqt_mean_1",
        "spectral_centroid_mean_1",
        "spectral_bandwidth_mean_1",
        "spectral_rolloff_mean_1",
        "rmse_mean_1",
        "zcr_mean_1",
    }
    assert required_keys.issubset(feats.keys())


def test_extract_features_rejects_silent_audio(tmp_path):
    sr = 22_050
    silence = np.zeros(sr // 4)
    silent_path = tmp_path / "silent.wav"
    sf.write(silent_path, silence, sr)
    with pytest.raises(ValueError):
        extract_features(silent_path)


def test_extract_many_walks_directories_and_skips_failures(tmp_path, tone_writer, capsys):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    tone_writer(filename="tone.wav")  # writes into tmp_path
    (tmp_path / "tone.wav").replace(audio_dir / "tone.wav")
    sf.write(audio_dir / "silent.wav", np.zeros(2_000), 22_050)

    df = extract_many([str(audio_dir)], pattern="*.wav")

    assert len(df) == 1
    assert df.loc[0, "source_path"].endswith("tone.wav")

    captured = capsys.readouterr()
    assert "Failed on" in captured.err


def test_cli_emits_csv(tmp_path, tone_writer):
    project_root = Path(__file__).resolve().parents[1]
    audio_path = tone_writer(filename="cli.wav")
    output_csv = tmp_path / "features.csv"

    cmd = [
        sys.executable,
        "scripts/extract_librosa_features.py",
        str(audio_path),
        "--output",
        str(output_csv),
    ]

    subprocess.run(cmd, cwd=project_root, check=True)

    df = pd.read_csv(output_csv)
    assert df.shape[0] == 1
    assert df.columns[0] == "source_path"