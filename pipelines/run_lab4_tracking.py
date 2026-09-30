"""Lab 4 pipeline with lightweight JSON experiment tracking."""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import (
    load_data,
    clean_data,
    build_feature_matrix,
    save_metadata,
    build_preference_data,
)

from train import train_recommenders, train_preference_models

import joblib


MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def progress(message):
    print(f"\n[LAB 4] {message}", flush=True)


start = time.time()


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

progress("Loading anime.csv and rating.csv...")
anime, ratings = load_data()

print(f"Anime rows: {len(anime):,}", flush=True)
print(f"Rating rows: {len(ratings):,}", flush=True)


# ------------------------------------------------------------
# 2. Clean data
# ------------------------------------------------------------

progress("Cleaning datasets...")
anime, ratings, rated = clean_data(
    anime,
    ratings
)

print(f"Rated rows: {len(rated):,}", flush=True)


# ------------------------------------------------------------
# 3. Build anime feature matrix
# ------------------------------------------------------------

progress("Building anime feature matrix...")
features, scaler = build_feature_matrix(anime)

print(f"Feature matrix shape: {features.shape}", flush=True)


# ------------------------------------------------------------
# 4. Train recommendation models
# ------------------------------------------------------------

progress("Training recommendation models...")
similarity, knn = train_recommenders(
    anime,
    features
)

progress("Recommendation models completed.")


# ------------------------------------------------------------
# 5. Save metadata
# ------------------------------------------------------------

progress("Saving dataset metadata...")
save_metadata(
    anime,
    ratings,
    rated,
    features
)


# ------------------------------------------------------------
# 6. Build preference data
# ------------------------------------------------------------

progress("Building user preference dataset...")
ml_data = build_preference_data(
    anime,
    rated
)

print(f"Preference dataset rows: {len(ml_data):,}", flush=True)


# ------------------------------------------------------------
# 7. Train preference models
# ------------------------------------------------------------

progress("Training Logistic Regression and Random Forest...")
trained, results = train_preference_models(
    ml_data
)

progress("Preference models completed.")


# ------------------------------------------------------------
# 8. Create directories
# ------------------------------------------------------------

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 9. Save recommendation models
# ------------------------------------------------------------

progress("Saving recommendation models...")

joblib.dump(
    anime,
    MODELS_DIR / "anime_cleaned.joblib"
)

joblib.dump(
    features,
    MODELS_DIR / "feature_matrix.joblib"
)

joblib.dump(
    similarity,
    MODELS_DIR / "cosine_similarity.joblib"
)

joblib.dump(
    knn,
    MODELS_DIR / "knn_recommender.joblib"
)

joblib.dump(
    scaler,
    MODELS_DIR / "numeric_scaler.joblib"
)


# ------------------------------------------------------------
# 10. Save preference models
# ------------------------------------------------------------

progress("Saving preference models...")

for name, model in trained.items():

    joblib.dump(
        model,
        MODELS_DIR / f"{name}.joblib"
    )


# ------------------------------------------------------------
# 11. Create run record
# ------------------------------------------------------------

duration = round(
    time.time() - start,
    3
)

run_record = {
    "pipeline": "lab4_tracking",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "duration_seconds": duration,
    "anime_rows": len(anime),
    "rating_rows": len(ratings),
    "rated_rows": len(rated),
    "feature_shape": list(features.shape),
    "preference_model_results": results,
}


# ------------------------------------------------------------
# 12. Save tracking log
# ------------------------------------------------------------

with open(
    LOGS_DIR / "run_history.jsonl",
    "a",
    encoding="utf-8"
) as f:

    f.write(
        json.dumps(run_record) + "\n"
    )


# ------------------------------------------------------------
# 13. Save results
# ------------------------------------------------------------

with open(
    OUTPUTS_DIR / "preference_model_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ------------------------------------------------------------
# 14. Final result
# ------------------------------------------------------------

progress("LAB 4 COMPLETED SUCCESSFULLY!")

print(
    json.dumps(
        run_record,
        indent=2
    )
)