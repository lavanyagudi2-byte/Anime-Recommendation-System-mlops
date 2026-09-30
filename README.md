# 🎌 Anime Recommendation System

A Machine Learning-based Anime Recommendation System developed using Python.  
The project analyzes anime and user-rating data and provides anime recommendations using **Cosine Similarity** and **K-Nearest Neighbors (KNN)**. It also predicts user preferences using **Logistic Regression** and **Random Forest**.

---

## 📌 Project Overview

The Anime Recommendation System is designed to recommend anime based on similarities between anime features and to predict whether a user is likely to like an anime.

The project includes:

- Data preprocessing and cleaning
- Exploratory data analysis
- Anime feature engineering
- Content-based recommendation
- Cosine Similarity
- K-Nearest Neighbors (KNN)
- User preference prediction
- Logistic Regression
- Random Forest
- Model evaluation
- Reproducible pipeline scripts

---
# 🎯 Project Objectives

The main objectives of this project are:

1. Load and understand anime and user-rating datasets.
2. Perform data cleaning and preprocessing.
3. Handle missing values.
4. Remove duplicate records.
5. Separate rated and unrated user records.
6. Perform feature engineering.
7. Convert categorical anime information into numerical features.
8. Standardize numerical features.
9. Build an anime feature matrix.
10. Calculate anime similarity using Cosine Similarity.
11. Build a KNN-based recommendation system.
12. Create a user preference classification dataset.
13. Train Logistic Regression.
14. Train Random Forest.
15. Evaluate classification models.
16. Save trained models using Joblib.
17. Save recommendation and evaluation results.
18. Convert the original notebook into a reusable project structure.

---

# 🧠 Machine Learning Components

The project contains two major Machine Learning components.

## Component 1: Anime Recommendation

A content-based recommendation approach is used.

Anime information is converted into numerical features and used to determine similarity between anime.

The recommendation system uses:

- Cosine Similarity
- K-Nearest Neighbors

---

## Component 2: User Preference Prediction

The second component predicts whether a user is likely to like an anime.

The target variable is created from the user's rating.

```text
Rating >= 7
      ↓
Liked = 1

Rating < 7
      ↓
Liked = 0
# 🎯 Objectives

The main objectives of this project are:

- Perform data preprocessing and cleaning.
- Analyze anime and user-rating data.
- Create useful numerical features from anime information.
- Build a content-based anime recommendation system.
- Implement Cosine Similarity.
- Implement K-Nearest Neighbors (KNN).
- Predict whether a user is likely to like an anime.
- Train Logistic Regression and Random Forest models.
- Evaluate recommendation and classification models.
- Convert the notebook-based project into an organized and reusable project structure.
- Create executable pipeline scripts for the project.

---

# 🧠 Machine Learning Approach

The project contains two main Machine Learning components.

## 1. Anime Recommendation

A content-based recommendation approach is used.

Anime features are created from:

- Genre
- Type
- Episodes
- Rating
- Members

Categorical features are converted using one-hot encoding, while numerical features are standardized using `StandardScaler`.

The resulting feature matrix is used to calculate similarity between anime.

### Recommendation techniques

- Cosine Similarity
- K-Nearest Neighbors

---

## 📂 Project Structure

```text
anime-recommendation/
│
├── data/
│   ├── raw/
│   │   ├── anime.csv
│   │   ├── rating.csv
│   │   └── README.md
│   │
│   └── processed/
│       └── dataset_metadata.json
│
├── notebooks/
│   └── project_implementation.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── pipelines/
│   ├── run_lab3_baseline.py
│   └── run_lab4_tracking.py
│
├── models/
│   └── Generated model files
│
├── outputs/
│   └── Evaluation and recommendation results
│
├── logs/
│
├── artifacts/
│
├── requirements.txt
├── README.md
└── .gitignore
