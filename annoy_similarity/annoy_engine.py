import joblib
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

class SpotifyAnnoyEngine:
    def __init__(self, model_dir="model/", metric="angular"):
        self.scaler = joblib.load(model_dir + "scaler.pkl")

        self.X = np.load(model_dir + "features.npy")

        self.meta = pd.read_csv(model_dir + "metadata.csv")

        f = self.X.shape[1]
        self.index = AnnoyIndex(f, metric)
        self.index.load(model_dir + "ann_index.ann")

        print("Spotify Annoy engine ready.")

    def query_by_track_index(self, idx, k=10):
        """Search by row index in the dataset."""
        vec = self.X[idx]
        ids = self.index.get_nns_by_vector(vec, k, include_distances=True)
        idxs, dists = ids
        return self.meta.iloc[idxs], dists

    def query_by_features(self, raw_features, k=10):
        """Search using new external features (e.g. uploaded audio)."""
        vec = self.scaler.transform([raw_features])[0]
        idxs, dists = self.index.get_nns_by_vector(vec, k, include_distances=True)
        return self.meta.iloc[idxs], dists

    def lookup(self, track_id):
        """Return dataset index for a specific track_id."""
        return self.meta.index[self.meta.track_id == track_id][0]
