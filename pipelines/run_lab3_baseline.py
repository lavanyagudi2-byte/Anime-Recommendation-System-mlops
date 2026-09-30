"""Lab 3 baseline pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import load_data, clean_data, build_feature_matrix, save_metadata
from train import train_recommenders

import joblib

MODELS_DIR = PROJECT_ROOT / "models"

anime, ratings = load_data()
anime, ratings, rated = clean_data(anime, ratings)
features, scaler = build_feature_matrix(anime)
similarity, knn = train_recommenders(anime, features)
save_metadata(anime, ratings, rated, features)

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(anime, MODELS_DIR / "anime_cleaned.joblib")
joblib.dump(features, MODELS_DIR / "feature_matrix.joblib")
joblib.dump(similarity, MODELS_DIR / "cosine_similarity.joblib")
joblib.dump(knn, MODELS_DIR / "knn_recommender.joblib")
joblib.dump(scaler, MODELS_DIR / "numeric_scaler.joblib")

print("Lab 3 baseline completed successfully.")
