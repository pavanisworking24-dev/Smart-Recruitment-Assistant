# Smart-Recruitment-Assistant

An AI-powered recruitment support system that scores, ranks, and segments job candidates using Machine Learning — helping HR teams make faster, data-driven hiring decisions.

Live Demo link - https://smart-recruitment-assistant-dw9sbvnpeopxabux9tvkas.streamlit.app/

---

## Project Overview

Manual candidate screening is time-consuming and inconsistent. This project builds an end-to-end intelligent recruitment pipeline that:

- **Predicts** the likelihood of a candidate being selected (binary classification)
- **Scores** each candidate with a 0–100 Hiring Score derived from calibrated model probabilities
- **Ranks** candidates within their job role so recruiters can prioritise the most suitable applicants
- **Segments** candidates into four quality tiers: Excellent, Strong, Average, and Needs Improvement
- **Explains** every score using SHAP values so recruiters understand the key factors behind each decision
- **Audits** predictions for potential proxy bias using held-out sensitive attributes

### Business Impact (Top 20% shortlist scenario)

| Metric | Result |
|---|---|
| Candidates reviewed | 20% of total applicant pool |
| True hires captured | 52% recall (2.6× lift over random screening) |
| Screening effort reduction | 80% fewer candidates to manually review |
| Estimated time saved | ~267 hours per hiring cycle* |

*Based on a documented assumption of 5 minutes per manual application review. See `06_business_impact_simulation.ipynb` for full methodology and sensitivity analysis.

---

## Job Roles Covered

| Role | Selection Rate | Key hiring signals |
|---|---|---|
| Software Engineer | ~25% | Technical skill, project quality, GitHub profile |
| Data Analyst | ~29% | Technical skill, aptitude, certifications |
| Sales Executive | ~38% | Communication, LinkedIn presence, relocation |
| Strategy Consultant | ~18% | Aptitude, education level, communication |

---

## Project Structure

```
smart_recruitment_assistant/
│
├── notebooks/
│   ├── 01_data_generation.ipynb           # Synthetic dataset generation (4,000 candidates, 18 features)
│   ├── 02_inspection_cleaning_eda.ipynb   # Data inspection, cleaning, and exploratory analysis
│   ├── 03_classification_modeling.ipynb   # Baseline → model comparison → tuning → calibration
│   ├── 04_hiring_score_and_ranking.ipynb  # Hiring Score generation and per-role ranking
│   ├── 05_clustering_shap_fairness.ipynb  # Clustering, SHAP explainability, fairness audit
│   └── 06_business_impact_simulation.ipynb # Quantified business value metrics
│
├── data/
│   ├── recruitment_candidates.csv          # Raw simulated dataset
│   ├── recruitment_candidates_cleaned.csv  # Post-cleaning dataset
│   └── recruitment_candidates_scored.csv   # Full dataset with Hiring Scores and ranks
│
├── outputs/
│   └── ranked_candidates.csv               # Recruiter-facing ranked shortlist
│
├── model/
│   └── final_classification_model.joblib   # Saved calibrated XGBoost model
│
├── app.py                                  # Streamlit web application
├── rebuild_model.py                        # Re-trains model locally (fixes sklearn version mismatch)
├── decision_log.md                         # Full reasoning trail for every project decision
└── README.md
```

---

## ML Pipeline Summary

### Data
- **4,000 synthetic candidates** across 4 job roles
- **18 features**: education, experience, technical skill, aptitude, communication, interview score, projects, certifications, ATS score, LinkedIn/GitHub profiles, relocation preference, and more
- **Realistic imperfections**: 3–5% missingness injected into 3 columns; ~1.5% outliers in `years_experience`
- **Sensitive attributes** (`age`, `gender`) held out of training; used only in the fairness audit

### Classification Model
| Model | CV ROC-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | 0.782 ± 0.027 | Linear model, good baseline |
| Random Forest | 0.772 ± 0.034 | Threshold-sensitive under class imbalance |
| **XGBoost (tuned + calibrated)** | **0.783 ± 0.020** | **Final model** |

Final model: XGBoost with GridSearch hyperparameter tuning (`max_depth=3`, `learning_rate=0.1`, `n_estimators=100`) and Platt scaling calibration. Brier score improved from 0.195 (uncalibrated) to 0.172 (calibrated).

### Clustering
KMeans (k=4) selected over Hierarchical and GMM based on silhouette score comparison. Cluster stability confirmed via Adjusted Rand Index (ARI ≥ 0.943 across 5 random seeds). Clusters labeled Excellent / Strong / Average / Needs Improvement based on centroid mean Hiring Score.

### SHAP Explainability
Global and per-role SHAP analysis validates the role-conditional feature weighting design — Software Engineer predictions weight technical skill and GitHub profile most heavily; Sales Executive weights communication; Strategy Consultant weights education level. Full analysis in `05_clustering_shap_fairness.ipynb`.

### Fairness Audit
Disparate impact ratios by gender: Female 0.986, Male 1.000, Other 0.867 — all above the standard 80% rule threshold. No proxy bias detected via relocation preference or age correlation checks. Limitation: the dataset was generated with sensitive attributes as statistically independent draws; the audit demonstrates methodology, not a guarantee of real-world fairness.

---

## Running the App

### Requirements

```bash
pip install streamlit pandas numpy joblib shap matplotlib xgboost scikit-learn
```

### If you get a scikit-learn version error on load

Re-train the model locally first (one-time, ~60 seconds):
```bash
python rebuild_model.py
```

### Launch

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**Mode 1 — Evaluate a Candidate**: input 17 candidate attributes → get Hiring Score, tier badge, rank within role, and a SHAP-based explanation of the top contributing factors.

**Mode 2 — Browse Ranked Pool**: filter by role, tier, or minimum score → view ranked leaderboard → download as CSV.

---

## Key Technical Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Data source | Fully simulated | Public recruitment datasets are too feature-poor; simulation grounded in real dataset structure with documented relationships |
| Model selection | XGBoost (tied with LR on AUC) | Preferred for downstream SHAP per-role analysis; tree structure captures role-specific interactions |
| Ranking approach | Within-role, not global | Different roles have different score distributions; cross-role comparison is invalid |
| Calibration | Platt scaling | Probabilities used directly as Hiring Score, so they must be trustworthy, not just useful for ranking |
| Class imbalance | class_weight + stratified splits | Role-varying selection rates (18–38%); accuracy is not a reliable metric here |
| Fairness audit | No bias injected, reported honestly | Injecting bias to then detect it would only test our own injection, not the audit methodology |

Full reasoning for every decision is documented in `decision_log.md`.

---

## Limitations

- Dataset is synthetic — results reflect simulated patterns, not real hiring data
- Business impact estimates depend on the 5-min/review assumption; substitute your own for a real deployment context
- Fairness audit demonstrates methodology; real proxy bias would require a deployment dataset with actual demographic correlations
- Model is trained and evaluated on the same distribution; production drift would require retraining

---

## Technologies Used

Python · XGBoost · scikit-learn · SHAP · Streamlit · pandas · NumPy · Matplotlib · Seaborn · Jupyter · joblib
