from flask import Flask, request, render_template, redirect, url_for
import sys
from pathlib import Path
import os

# --- Project Path Fix ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Model Imports ---
from scripts.download_songs import download_audio
from scripts.extract_librosa_features import extract_features as extract_features_annoy
from scripts.extract_librosa_features_GTZAN import extract_features as extract_features_GTZAN
from scripts.recommendation_service import RecommendationService
from genre_model.inference import GenreModel

# --- Global temp cache for loading page ---
cached_user_input = None
cached_file_bytes = None
cached_filename = None

# Tailwind Gradient List
gradients = [
    "from-pink-500/60 to-purple-600/40",
    "from-blue-500/60 to-indigo-600/40",
    "from-green-500/60 to-emerald-600/40",
    "from-yellow-500/60 to-amber-600/40",
    "from-orange-500/60 to-red-600/40"
]

# Flask App
app = Flask(__name__)

DEBUG_MODE = False

# FIXED: Use your working relative paths
recommender = RecommendationService(model_dir="../annoy_similarity/model/")

# Make classifier optional - won't crash if model file is missing (good feature from teammate!)
try:
    classifier = GenreModel(artifact_dir="../genre_model/artifacts")
    CLASSIFIER_AVAILABLE = True
    print("✅ Genre classifier loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠️  Genre classifier model not found: {e}")
    print("⚠️  Genre prediction will be disabled. App will continue without it.")
    classifier = None
    CLASSIFIER_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Error loading genre classifier: {e}")
    print("⚠️  Genre prediction will be disabled. App will continue without it.")
    classifier = None
    CLASSIFIER_AVAILABLE = False

DEBUG_AUDIO_PATH = PROJECT_ROOT / "flask-website-demo" / "test.wav"
DEBUG_EMBED_URL = "https://www.youtube.com/embed/dQw4w9WgXcQ"


# -------------------------------------------------------
# ----------------------- ROUTES ------------------------
# -------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Upload page"""
    return render_template("index.html")


@app.route("/", methods=["POST"])
def index_post():
    """Handle submit → save data → redirect to loading page"""
    global cached_user_input, cached_file_bytes, cached_filename

    user_input = request.form.get("user_input")
    file = request.files.get("audio_file")

    cached_user_input = user_input

    # If file uploaded → save bytes before redirect (important!)
    if file and file.filename != "":
        cached_file_bytes = file.read()      # RAW BYTES — SAFE!
        cached_filename = file.filename
    else:
        cached_file_bytes = None
        cached_filename = None

    return redirect(url_for("loading"))


@app.route("/loading")
def loading():
    """Shows loading animation page"""
    return render_template("loading.html")


@app.route("/results")
def results_page():
    """Actual processing happens here AFTER the loading page."""
    global cached_user_input, cached_file_bytes, cached_filename

    # ---- CASE 1: FILE UPLOAD ----
    if cached_file_bytes:
        # Use relative path that works from flask-website-demo directory
        upload_path = Path("uploads") / "uploaded_audio.wav"
        upload_path.parent.mkdir(exist_ok=True)

        # reconstruct file from bytes
        with open(upload_path, "wb") as f:
            f.write(cached_file_bytes)

        # Process file normally
        annoy_features = extract_features_annoy(upload_path)
        GTZAN_features = extract_features_GTZAN(upload_path)

        result_df = recommender.recommend_from_features(annoy_features, k=10)
        results = result_df.to_dict(orient="records")

        # Hearts
        for r in results:
            accuracy = r["distance"] * 100
            full = 10 - int(accuracy / 10)
            half = 1 if float(accuracy - full) >= 0.5 else 0
            empty = 10 - full - half
            r["hearts"] = {"full": full, "half": half, "empty": empty}

        # Genre - handle missing classifier (good feature from teammate!)
        if CLASSIFIER_AVAILABLE:
            genre_list = classifier.top_k(GTZAN_features, k=1)
            top_genre = genre_list[0]["genre"] if genre_list else "Unknown"
        else:
            top_genre = "Genre Detection Unavailable"

        upload_path.unlink()
        return render_template(
            "results.html",
            results=results,
            genre=top_genre,
            embed_url=None,
            gradients=gradients
        )

    # ---- CASE 2: LINK INPUT ----
    if not cached_user_input:
        return render_template("index.html", error="Please paste a link or upload a file.")

    if DEBUG_MODE:
        downloaded_path = DEBUG_AUDIO_PATH
        embed_url = DEBUG_EMBED_URL
    else:
        download = download_audio(cached_user_input)
        downloaded_path = download[0]
        embed_url = download[1]

        if downloaded_path is None:
            return render_template("index.html", error="Could not download audio.")

    try:
        annoy_features = extract_features_annoy(downloaded_path)
        GTZAN_features = extract_features_GTZAN(downloaded_path)

        result_df = recommender.recommend_from_features(annoy_features, k=10)
        results = result_df.to_dict(orient="records")

        # Hearts
        for r in results:
            accuracy = r["distance"] * 100
            full = 10 - int(accuracy / 10)
            half = 1 if float(accuracy - full) >= 0.5 else 0
            empty = 10 - full - half
            r["hearts"] = {"full": full, "half": half, "empty": empty}

        # Genre - handle missing classifier (good feature from teammate!)
        if CLASSIFIER_AVAILABLE:
            genre_list = classifier.top_k(GTZAN_features, k=1)
            top_genre = genre_list[0]["genre"] if genre_list else "Unknown"
        else:
            top_genre = "Genre Detection Unavailable"

    finally:
        if not DEBUG_MODE and downloaded_path.exists():
            downloaded_path.unlink()

    return render_template(
        "results.html",
        results=results,
        genre=top_genre,
        embed_url=embed_url,
        gradients=gradients
    )


# Run App
if __name__ == "__main__":
    app.run(debug=True)