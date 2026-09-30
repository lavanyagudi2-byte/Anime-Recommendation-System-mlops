"""Training functions for anime recommendation and preference prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess import (
    load_data,
    clean_data,
    build_feature_matrix,
    build_preference_data,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# RECOMMENDATION MODELS
# ============================================================

def train_recommenders(anime, feature_matrix):
    """Train cosine similarity and KNN recommendation models."""

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)

    # KNN recommender
    knn = NearestNeighbors(
        n_neighbors=11,
        metric="cosine"
    )

    knn.fit(feature_matrix)

    return similarity_matrix, knn


# ============================================================
# USER PREFERENCE PREDICTION
# ============================================================

def train_preference_models(ml_data):
    """Train Logistic Regression and Random Forest models."""

    # Use a manageable sample for faster training
    if len(ml_data) > 500000:
        ml_data = ml_data.sample(
            n=500000,
            random_state=42
        )
    # Features used for prediction
    features = [
        "rating_anime",
        "members",
        "user_avg_rating",
        "user_rating_count",
    ]

    # Select features
    X = ml_data[features].copy()

    # Convert features to numeric
    for col in features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using median
    X = X.fillna(
        X.median()
    )

    # Target variable
    y = ml_data["liked"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Models
    models = {
        "logistic_regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                ),
            ]
        ),

        "random_forest": RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    n_jobs=-1,
),
        
    }

    # Store trained models
    trained = {}

    # Store evaluation results
    results = []

    # Train each model
    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        # Predictions
        pred = model.predict(
            X_test
        )

        # Evaluation metrics
        result = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                pred
            ),
            "Precision": precision_score(
                y_test,
                pred,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                pred,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                pred,
                zero_division=0
            ),
        }

        results.append(result)

        trained[name] = model

    return trained, results


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # Create models directory
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load datasets
    anime, ratings = load_data()

    # Clean datasets
    anime, ratings, rated = clean_data(
        anime,
        ratings
    )

    # Build feature matrix
    feature_matrix, scaler = build_feature_matrix(
        anime
    )

    # Train recommendation models
    similarity_matrix, knn = train_recommenders(
        anime,
        feature_matrix
    )

    # Save recommendation models
    joblib.dump(
        anime,
        MODELS_DIR / "anime_cleaned.joblib"
    )

    joblib.dump(
        feature_matrix,
        MODELS_DIR / "feature_matrix.joblib"
    )

    joblib.dump(
        similarity_matrix,
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

    # Build preference prediction dataset
    ml_data = build_preference_data(
        anime,
        rated
    )

    # Train preference models
    trained_models, results = train_preference_models(
        ml_data
    )

    # Save preference models
    for name, model in trained_models.items():

        joblib.dump(
            model,
            MODELS_DIR / f"{name}.joblib"
        )

    # Display results
    print()
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Preference Model Results:")

    for result in results:
        print()
        print(f"Model: {result['Model']}")
        print(f"Accuracy:  {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall:    {result['Recall']:.4f}")
        print(f"F1 Score:  {result['F1 Score']:.4f}")

    print()
    print("Models saved in:")
    print(MODELS_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()