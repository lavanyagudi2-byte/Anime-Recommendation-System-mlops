"""Evaluation and recommendation utilities."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def find_anime_index(anime, anime_name):
    matches = anime[anime["name"].astype(str).str.lower() == anime_name.lower()]
    if matches.empty:
        return None
    return matches.index[0]


def recommend_cosine(anime, similarity_matrix, anime_name, n=10):
    index = find_anime_index(anime, anime_name)
    if index is None:
        raise ValueError(f"Anime not found: {anime_name}")

    scores = similarity_matrix[index]
    similar_indices = np.argsort(scores)[::-1][1:n + 1]

    result = anime.iloc[similar_indices][
        ["name", "genre", "type", "episodes", "rating", "members"]
    ].copy()
    result["similarity"] = scores[similar_indices]
    return result


def recommend_knn(anime, knn, feature_matrix, anime_name, n=10):
    index = find_anime_index(anime, anime_name)
    if index is None:
        raise ValueError(f"Anime not found: {anime_name}")

    distances, indices = knn.kneighbors(
        feature_matrix.iloc[[index]], n_neighbors=n + 1
    )
    result_indices = indices[0][1:]
    result_distances = distances[0][1:]

    result = anime.iloc[result_indices][
        ["name", "genre", "type", "episodes", "rating", "members"]
    ].copy()
    result["similarity"] = 1 - result_distances
    return result


def evaluate_recommenders(anime_name="Naruto", n=10):
    anime = joblib.load(MODELS_DIR / "anime_cleaned.joblib")
    features = joblib.load(MODELS_DIR / "feature_matrix.joblib")
    cosine = joblib.load(MODELS_DIR / "cosine_similarity.joblib")
    knn = joblib.load(MODELS_DIR / "knn_recommender.joblib")

    cosine_result = recommend_cosine(anime, cosine, anime_name, n)
    knn_result = recommend_knn(anime, knn, features, anime_name, n)

    comparison = pd.DataFrame({
        "Model": ["Cosine Similarity", "KNN"],
        "Average Similarity": [
            cosine_result["similarity"].mean(),
            knn_result["similarity"].mean(),
        ],
    })

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cosine_result.to_csv(OUTPUTS_DIR / "cosine_recommendations.csv", index=False)
    knn_result.to_csv(OUTPUTS_DIR / "knn_recommendations.csv", index=False)
    comparison.to_csv(OUTPUTS_DIR / "recommendation_comparison.csv", index=False)

    return comparison


if __name__ == "__main__":
    print(evaluate_recommenders())
