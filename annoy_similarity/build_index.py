import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from annoy import AnnoyIndex

# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "../data/processed/fma_metadata_features_joined.csv"
ID_COL = "track_id"
TITLE_COL = "track_title"
ARTIST_COL = "artist_name"
ALBUM_COL = "album_title"

FEATURE_PREFIXES = [
    "mfcc_",
    "chroma_cqt_",
    "spectral_",
    "rmse_",
    "zcr_"
]

MODEL_DIR = "annoy_similarity/model/"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)

feature_cols = [
    c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)
]

print("Feature columns:", len(feature_cols))

X = df[feature_cols].fillna(0).values

# ------------------------------
# SCALE FEATURES
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, MODEL_DIR + "scaler.pkl")
np.save(MODEL_DIR + "features.npy", X_scaled)

df[[ID_COL, TITLE_COL, ARTIST_COL, ALBUM_COL]].to_csv(MODEL_DIR + "metadata.csv", index=False)

# ------------------------------
# BUILD ANNOY INDEX
# ------------------------------
f = X_scaled.shape[1]
index = AnnoyIndex(f, "angular")

print("Building Annoy index...")
for i, row in enumerate(X_scaled):
    index.add_item(i, row)

index.build(50) 
index.save(MODEL_DIR + "ann_index.ann")

print("Done.")
