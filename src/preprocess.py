"""Preprocessing utilities for the Anime Recommendation System."""

from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_data():
    anime_path = RAW_DIR / "anime.csv"
    rating_path = RAW_DIR / "rating.csv"
    if not anime_path.exists() or not rating_path.exists():
        raise FileNotFoundError(
            "Place anime.csv and rating.csv in data/raw/ before running the pipeline."
        )
    return pd.read_csv(anime_path), pd.read_csv(rating_path)


def clean_data(anime, ratings):
    anime = anime.drop_duplicates().copy()
    ratings = ratings.drop_duplicates().copy()

    rated = ratings[ratings["rating"] != -1].copy()

    for col in ["genre", "type"]:
        anime[col] = anime[col].fillna("Unknown")

    for col in ["episodes", "rating", "members"]:
        anime[col] = pd.to_numeric(anime[col], errors="coerce")
        anime[col] = anime[col].fillna(anime[col].median())

    return anime, ratings, rated


def build_feature_matrix(anime):
    genre_matrix = anime["genre"].str.get_dummies(sep=", ")
    type_matrix = pd.get_dummies(anime["type"], prefix="type")

    numeric_features = anime[["episodes", "rating", "members"]]
    scaler = StandardScaler()
    numeric_matrix = pd.DataFrame(
        scaler.fit_transform(numeric_features),
        columns=numeric_features.columns,
        index=anime.index,
    )

    feature_matrix = pd.concat(
        [genre_matrix, type_matrix, numeric_matrix], axis=1
    )

    return feature_matrix, scaler


def build_preference_data(anime, rated):
    ml_data = rated.copy()
    ml_data["liked"] = (ml_data["rating"] >= 7).astype(int)

    ml_data = ml_data.merge(
        anime[["anime_id", "rating", "members"]],
        on="anime_id",
        how="left",
        suffixes=("_user", "_anime"),
    )

    user_stats = (
        rated.groupby("user_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "user_avg_rating",
                "count": "user_rating_count",
            }
        )
    )

    return ml_data.merge(user_stats, on="user_id", how="left")


def save_metadata(anime, ratings, rated, feature_matrix):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "anime_rows": int(len(anime)),
        "anime_columns": int(anime.shape[1]),
        "rating_rows": int(len(ratings)),
        "rated_rows": int(len(rated)),
        "feature_rows": int(feature_matrix.shape[0]),
        "feature_columns": int(feature_matrix.shape[1]),
    }
    (PROCESSED_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
