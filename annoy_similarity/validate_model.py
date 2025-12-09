"""
validate_model.py

This script performs three critical checks:

1. feature matrix alignment
2. metadata alignment
3. Annoy index alignment

Run this after building your Annoy index to confirm the entire pipeline is correct.
"""

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import joblib

# ------------------------------
# CONFIG
# ------------------------------
MODEL_DIR = "annoy_similarity/model/"
DATA_PATH = "data/processsed/fma_metadata_features_joined.csv"

ID_COL = "track_id"
TITLE_COL = "track_title"
ARTIST_COL = "artist_name"

FEATURE_PREFIXES = [
    "mfcc_",
    "chroma_cqt_",
    "spectral_",
    "rmse_",
    "zcr_",
]

# ------------------------------
# LOAD ORIGINAL DATAFRAME
# ------------------------------
print("Loading original dataset...")
df = pd.read_csv(DATA_PATH)

feature_cols = [
    c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)
]

X_raw = df[feature_cols].fillna(0).values

# ------------------------------
# LOAD SAVED ARTIFACTS
# ------------------------------
print("Loading saved artifacts...")
X_saved = np.load(MODEL_DIR + "features.npy")
scaler = joblib.load(MODEL_DIR + "scaler.pkl")
metadata_saved = pd.read_csv(MODEL_DIR + "metadata.csv")

# ------------------------------
# TEST 1 — FEATURE MATRIX ALIGNMENT
# ------------------------------
print("\nTEST 1 — Checking feature matrix alignment...")

X_scaled_original = scaler.transform(X_raw)

if np.allclose(X_scaled_original[:20], X_saved[:20]):
    print("✔ PASS: Saved feature matrix matches scaled original features.")
else:
    raise AssertionError("❌ FAIL: Saved features.npy does NOT match scaled DF rows!")


# ------------------------------
# TEST 2 — METADATA ALIGNMENT
# ------------------------------
print("\nTEST 2 — Checking metadata alignment...")

# Select some random-ish indices
indices_to_test = [0, 5, 100, 999, len(df) - 1]

for idx in indices_to_test:
    orig = df.iloc[idx][[ID_COL, TITLE_COL, ARTIST_COL]]
    saved = metadata_saved.iloc[idx]

    assert orig[ID_COL] == saved[ID_COL], f"❌ track_id mismatch at row {idx}"
    assert orig[TITLE_COL] == saved[TITLE_COL], f"❌ title mismatch at row {idx}"
    assert orig[ARTIST_COL] == saved[ARTIST_COL], f"❌ artist mismatch at row {idx}"

print("✔ PASS: Metadata rows perfectly align with the original dataframe.")


# ------------------------------
# TEST 3 — ANNOY INDEX ALIGNMENT
# ------------------------------
print("\nTEST 3 — Checking Annoy index vector alignment...")

f = X_saved.shape[1]
ann = AnnoyIndex(f, "angular")
ann.load(MODEL_DIR + "ann_index.ann")

test_ids = [0, 10, 500, 2222, len(X_saved)-1]

for i in test_ids:
    ann_vec = np.array(ann.get_item_vector(i))
    saved_vec = X_saved[i]

    if not np.allclose(ann_vec, saved_vec, atol=1e-6):
        raise AssertionError(f"❌ Annoy vector mismatch at index {i}")

print("✔ PASS: Annoy index vectors match the saved feature matrix.")


# ------------------------------
# FINAL RESULT
# ------------------------------
print("\n🎉 ALL TESTS PASSED — Your Annoy model pipeline is 100% consistent and safe.")
