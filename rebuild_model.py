"""
rebuild_model.py
----------------
Run this script ONCE in your project folder to rebuild final_classification_model.joblib
using your local scikit-learn version, fixing the _RemainderColsList compatibility error.

Usage:
    pip install pandas scikit-learn xgboost joblib
    python rebuild_model.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

print("Loading cleaned dataset...")
df = pd.read_csv("recruitment_candidates_cleaned.csv")

feature_cols = [
    "job_role", "education_level", "years_experience", "technical_skill_score",
    "aptitude_score", "communication_score", "interview_score", "internship_experience",
    "projects_count", "project_quality_score", "certifications_count",
    "certification_prestige_score", "competition_awards_count", "ats_score",
    "linkedin_profile_score", "github_coding_profile_score", "relocation_preference"
]
cat_cols = ["job_role", "education_level", "relocation_preference"]

X = df[feature_cols]
y = df["selected"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    ))
])

print("Tuning XGBoost with GridSearch (this takes ~30-60 seconds)...")
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}
gs = GridSearchCV(xgb_pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
best_xgb = gs.best_estimator_

print(f"Best params: {gs.best_params_}")
print(f"Best CV ROC-AUC: {gs.best_score_:.3f}")

print("Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(best_xgb, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
proba = calibrated_model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC (calibrated): {roc_auc_score(y_test, proba):.3f}")

print("Saving model...")
joblib.dump(calibrated_model, "final_classification_model.joblib")
print("Done — final_classification_model.joblib saved successfully.")
print("You can now run: streamlit run app.py")
