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
