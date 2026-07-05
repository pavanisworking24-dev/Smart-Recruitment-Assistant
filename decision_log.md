# Smart Recruitment Assistant — Project Decision Log

> Living document. Updated as decisions are made. Use this as the source of truth for "why did we do X" when explaining the project (resume, interviews, report).

---

## Project Summary

**Goal:** Build an AI-powered Smart Recruitment Assistant that scores, ranks, and segments job candidates using ML, to support data-driven hiring decisions.

**Scope level chosen:** Advanced
**Reason:** Prior experience with advanced-level ML projects; comfortable with the added complexity (multi-role modeling, tuning, explainability, fairness audit, deployment).

**Core ML tasks (3 distinct problems, explicitly separated):**
1. **Classification** — predict Selected / Rejected
2. **Ranking** — order candidates within their job role by Hiring Score
3. **Clustering** — unsupervised segmentation into Excellent / Strong / Average / Needs Improvement

---

## Decision Log

### D1 — Data Source Strategy
- **Question:** Use real data, find a dataset, or simulate?
- **Options considered:**
  - A. Real Kaggle dataset only, narrow scope to match it
  - B. Real dataset + simulate extra features on top
  - C. Fully simulated dataset, using real dataset(s) as a structural reference
- **Searched:** Kaggle "Predicting Hiring Decisions in Recruitment Data" (rabieelkharoua) — 1,501 samples, 10 features, binary Hired/Not Hired label. Found to be too narrow/limited in feature richness for our brief.
- **Decision:** **Option C.** Fully simulate the dataset, using the real Kaggle dataset as a structural/sanity-check reference (value ranges, realistic relationships) — not copied data.
- **Why defensible:** Public recruitment datasets are scarce, small, and feature-poor relative to the brief's requirements (education, skills, aptitude, communication, certs, interviews, projects, experience all together). Synthetic data is a legitimate and common choice when designed with documented, logical feature relationships rather than arbitrary randomness.

### D2 — Project Scope Tier
- **Question:** Basic / Intermediate / Advanced scope?
- **Decision:** **Advanced** — full pipeline including hyperparameter tuning, model calibration, SHAP explainability, fairness/bias audit, multi-role modeling, and Streamlit deployment.

### D3 — Ranking Across Multiple Job Roles
- **Question:** With multiple job roles in the dataset, how do we rank candidates? Globally, or per role?
- **Problem identified:** Different roles value different attributes (e.g., technical skill matters more for Software Engineer, communication matters more for Sales). A single global ranking would invalidly compare candidates across incompatible criteria.
- **Options considered:**
  - A. Separate model trained per job role
  - B. One global model with `job_role` as an input feature; ranking computed within role groups
- **Decision:** **Option B.** One shared model (tree-based, e.g. Random Forest/XGBoost) that takes `job_role` as a feature and naturally learns role-specific interactions. Final ranking step always does `groupby('job_role')` before sorting by Hiring Score — candidates are only ever ranked against others who applied for the same role.
- **Why defensible:** More training data per model (vs. splitting data 4 ways), simpler pipeline, and tree-based models naturally capture feature-interaction effects across roles. Validated later via per-role SHAP analysis (confirming the model actually learned different important features per role).

### D4 — Job Roles Selected
- **Question:** Which 4th job role to add (beyond Data Analyst, Software Engineer, Sales Executive)?
- **Options considered:** Full Stack Developer, Project Manager, Strategy Consultant
- **Decision:** **Strategy Consultant.**
- **Why:** Full Stack Developer overlaps too heavily with Software Engineer (both technical, weak contrast). Strategy Consultant adds genuine contrast — leans on aptitude/reasoning, education pedigree, and communication, with technical_skill_score mattering far less. This spread (technical ↔ interpersonal ↔ analytical) makes the "different roles value different features" story much stronger for the SHAP-by-role analysis later.
- **Final role set:** Data Analyst, Software Engineer, Sales Executive, Strategy Consultant

### D5 — Education Level: Granularity and Role-Conditioning
- **Question:** Should "High School" be removed/replaced (e.g., with Bachelor's-1/Bachelor's-2 by duration), since a degree seems necessary for these roles?
- **Initial idea:** Replace High School with Bachelor's split by program duration (3yr/4yr).
- **Pushback:** Degree *duration* isn't a meaningful qualification signal (it's a structural/regional artifact, not a competence signal) — encoding it ordinally would wrongly imply 4yr > 3yr.
- **Counter-point raised:** Is it even realistic to keep "High School" as an option if these roles require degrees?
- **Resolution:** Checked the claim against all 4 roles — it's **role-dependent**, not universal:
  - Strategy Consultant: degree is a near-hard requirement
  - Software Engineer / Data Analyst: degree usually expected, but self-taught/non-degree paths realistically exist (especially if compensated by strong skills/certs/projects)
  - Sales Executive: degree is **not** a strong requirement in reality
- **Decision:** Keep `education_level` as **High School → Bachelor's → Master's → PhD** (real ordinal signal), but sample it **conditionally on `job_role`** — i.e., different probability distributions per role (e.g., Strategy Consultant pool rarely includes High School; Sales Executive pool includes it commonly).
- **Why defensible:** More realistic than a flat universal rule; mirrors how real hiring markets actually differ by role. Also strengthens the later fairness audit (education-level access is a real equity axis worth examining).

### D6 — Additional Features (expanding from 12 → 17)
- **Question:** Add richer features beyond the original baseline?
- **Proposed additions and verdicts:**

| Feature | Verdict | Reasoning |
|---|---|---|
| `ats_score` | **Added** | Resume-keyword-match signal (distinct from raw competence) — realistic, and creates an interesting tension where skill and ATS score can diverge |
| `project_quality_score` (alongside `projects_count`) | **Added** | Quantity ≠ quality; depth/impact matters separately from count |
| `certification_prestige_score` (alongside `certifications_count`) | **Added** | Issuer tier matters (AWS/Microsoft/Google vs generic platforms) — kept separate from count so SHAP can reveal which matters more |
| `competition_awards_count` (tightened from vague "achievements") | **Added, flagged as weakest** | "Achievements" was too vague to be defensible; narrowed to concrete competition/hackathon/case-comp awards. Flagged as most likely to show weak/near-noise signal once data is generated — acceptable, becomes an evidence-based pruning candidate later |
| `linkedin_profile_score` | **Added** | Realistic professional-presence signal, especially relevant for Sales/Consulting roles |
| `github_coding_profile_score` | **Added, flagged for redundancy check** | Strong signal for SWE/Data Analyst, intentionally near-noise for Sales/Consulting (role-conditional relevance, mirrors the education-level approach) |

- **Final feature count:** 17 features + 2 held-out sensitive attributes (`age`, `gender` — excluded from training, used only in the fairness audit) + target (`selected`)

### D7 — Feature Redundancy / Removal Check
- **Question:** Should any of the 17 features be cut now, before data simulation?
- **Redundancy risks identified:**
  - `projects_count` vs `project_quality_score` (likely correlated)
  - `certifications_count` vs `certification_prestige_score` (likely correlated)
  - `technical_skill_score` vs `github_coding_profile_score` (conceptually close, higher risk)
- **Decision:** **Keep all 17 features for now.** Do not remove anything on a pre-data hunch.
- **Why:** Cutting features before generating data is guesswork, not analysis. Plan is to generate the data with intentionally injected independent noise (e.g., a skilled engineer who maintains a weak GitHub profile), then run a **correlation / multicollinearity (VIF) check** after EDA. Any removal will be evidence-based and documented as its own decision — a stronger resume/interview story than "I picked features that felt right."
- **Status:** Pending — revisit after data generation + EDA.

### D9 — Additional Feature: Relocation Preference
- **Question:** Add a feature for willingness to relocate?
- **Decision:** **Added.** `relocation_preference` — categorical: Yes / No / Open to Remote.
- **Why:** Realistic ATS/application field; fills a genuine gap (no existing feature touched geography/mobility); naturally role-conditional (near-zero importance for SWE/Data Analyst given 2026 remote-normalization, Medium for Sales due to regional territories, High for Strategy Consultant given the field's well-known travel/relocation expectations).
- **Bonus:** Flagged as a strong candidate for the later fairness audit — relocation willingness can act as a proxy for unrelated life-circumstance factors (caregiving, financial flexibility, age), making it a realistic source of indirect bias worth testing for, even though it's a "legitimate" business factor on its face.
- **Final feature count:** 18 features + 2 held-out sensitive attributes (`age`, `gender`) + target (`selected`)

### D10 — Role-Weight Table (Finalized)
- **Process:** Built an initial draft table (feature category × job role → Low/Medium/High/Very High importance), then revised twice based on direct pushback before locking.
- **Revisions made and reasoning:**
  - Communication: Software Engineer raised Low→Medium (engineers do standups, design docs, stakeholder presentations — "engineers don't need to communicate" was a stereotype, not reality)
  - Communication: Data Analyst raised Medium→High (translating data into decisions for stakeholders is core to the role)
  - Certifications: Data Analyst raised Medium→High (analytics is certification-driven in absence of one universal degree pathway — BI tools, SQL, cloud data certs)
  - GitHub profile: Data Analyst raised Medium→High, then negotiated down to **Medium** after discussion — reasoning: full High would erase the intended contrast with Software Engineer (the feature's whole purpose is showing role-conditional relevance — strong for SWE, weaker for Analyst, near-zero for Sales/Consulting); landed on Medium to preserve that gradient while still acknowledging GitHub matters somewhat for analysts (portfolio notebooks, SQL scripts)
  - LinkedIn profile: Software Engineer & Data Analyst raised Low→Medium (LinkedIn is now heavily used in technical recruiting too, not just for client-facing roles)
- **Final table:**

| Feature category | Software Engineer | Data Analyst | Sales Executive | Strategy Consultant |
|---|---|---|---|---|
| Technical skill | High | High | Low | Low |
| Aptitude | Medium | High | Low | Very High |
| Communication | Medium | High | Very High | High |
| Interview score | High | High | High | High |
| Experience | Medium | Medium | Medium | Medium |
| Education level | Medium | Medium | Low | Very High |
| Certifications | Medium | High | Low | Low |
| Projects | High | High | Low | Low |
| GitHub profile | High | Medium | Near-zero | Near-zero |
| LinkedIn profile | Medium | Medium | High | Medium |
| Relocation preference | Near-zero | Near-zero | Medium | High |
| ATS score | Medium (flat across all roles) |

- **Status:** Locked. These become literal numeric coefficients in the label-generation formula (Section: Data Simulation Logic).

### D11 — Selection Rate (Class Balance)
- **Question:** What % of candidates should be labeled `selected`? Flat rate across roles, or role-varying?
- **Options discussed:**
  - Flat ~25-30% for all roles (simpler, easier to model, less imbalance-handling overhead)
  - Role-varying, reflecting real differences in market/role competitiveness
- **Decision:** **Role-varying, centered around 20-30% overall.**
  - Strategy Consultant: ~15-20% selected (notoriously competitive, low acceptance, mirrors real consulting hiring funnels)
  - Software Engineer: ~25% selected
  - Data Analyst: ~28-30% selected
  - Sales Executive: ~35-40% selected (commonly hires more liberally, higher turnover roles)
- **Why defensible:** A single flat rate across four very different job markets would be less realistic. Role-varying rates also mean the clustering step's "Excellent" threshold will naturally differ by role — itself a genuine, demonstrable insight ("the bar for Excellent differs by role, reflecting real market competitiveness") rather than an artifact of the simulation.
- **Trade-off accepted:** Real class imbalance (especially for Strategy Consultant ~15-20%) means imbalance-handling techniques (class weights, stratified splits, F1/ROC-AUC over raw accuracy) are required in modeling — accepted as appropriate for Advanced scope, and as a deliberate demonstration of avoiding the "always predict majority class" trap.

### D13 — Data Generation: Implementation Notes & Fix
- **Built:** `01_data_generation.ipynb` — full pipeline implementing D1-D11's design (role distribution → role-conditional education → sensitive attributes → experience → weight table → core scores → interview score → supporting features → fit_score → label → missingness/outliers → save).
- **Issue found during build:** Naively setting the sigmoid threshold at the raw percentile of `fit_score` (e.g., "top 18% of scores" for Strategy Consultant) overshot the target selection rate, because `sigmoid(0) = 0.5` — candidates below the percentile threshold still had a real chance of being drawn as selected. First test run produced 33-42% selected across roles vs. ~18-38% targets.
- **Fix:** Replaced the percentile-based threshold with a binary search that finds the threshold whose *expected* selection rate (mean probability across the role, not just one random draw) matches the target exactly.
- **Verified results (4,000 candidates):**
  - Selection rates: Strategy Consultant 19.4%, Software Engineer 25.8%, Data Analyst 30.3%, Sales Executive 37.8%, overall 28.6% — matches D11 targets.
  - Baseline sanity check (untuned Logistic Regression): Accuracy 0.70, **ROC-AUC 0.77**, F1 0.58. This confirms the simulated problem is learnable but not trivially easy — exactly the target behavior from D3 (a hard cutoff would have produced near-perfect, unrealistic separability).
  - Missingness and outliers injected as planned (≈3-5% missing on `linkedin_profile_score`, `competition_awards_count`, `certification_prestige_score`; ~1.5% experience outliers).
- **Status:** Data generation complete. Dataset saved as `recruitment_candidates.csv` (4,000 rows × 21 columns: 17 predictive features + `job_role` + 2 held-out sensitive attributes + target).

### D15 — Notebook Structure: Combined Inspection + Cleaning + EDA
- **Question:** Separate notebooks for inspection/cleaning vs. EDA, or combine into one?
- **Discussion:** Methodologically, Inspection → Cleaning → EDA are three distinct phases.
  On a real-world dataset, inspection is genuine discovery; on our self-simulated dataset,
  it's verification of imperfections we deliberately injected (Decision D13/Section 11 of
  generation notebook) — a meaningful difference worth being able to articulate.
- **Decision:** Combine into a single notebook (`02_inspection_cleaning_eda.ipynb`), but
  preserve clear internal section boundaries (Part A — Inspection, Part B — Cleaning,
  Part C — EDA) so the methodological distinction stays visible and explainable even
  though it's one file.

### D16 — Missing Value & Outlier Handling Strategy
- **Missingness found:** `certification_prestige_score` (120 missing), `competition_awards_count` (200 missing), `linkedin_profile_score` (160 missing) — matches what was deliberately injected in Decision D13.
- **Strategy decided:** Median imputation for all three. Justified because missingness here is unconditional/MCAR by construction (injected independent of any other column) — no hidden pattern exists to model, so simple imputation is defensible. Median chosen over mean for robustness to skew/outliers in these columns.
- **Outlier handling:** IQR method flagged 120 outliers in `years_experience` (3.0% of data) — more than the 60 deliberately injected, because natural right-skew in a realistic experience distribution also pushes some legitimately high-experience candidates past the IQR fence. **Decision: cap (winsorize) at the IQR upper bound rather than drop rows** — avoids discarding otherwise-valid candidate data over one extreme field, consistent with how a real pipeline would handle a single suspicious value on an application.
- **Output:** Cleaned dataset saved separately as `recruitment_candidates_cleaned.csv`, so later notebooks (modeling, clustering, SHAP) start from a known-clean source.

### D17 — D7 Resolved: Feature Redundancy Check (Evidence-Based)
- **Question (deferred from D7):** Are `projects_count`/`project_quality_score`, `certifications_count`/`certification_prestige_score`, or `technical_skill_score`/`github_coding_profile_score` too redundant to keep both?
- **Evidence gathered:**
  - Pairwise correlations: 0.47, 0.45, and 0.44 respectively — moderate, not severe (concern threshold is typically ~0.8+).
  - Full VIF (Variance Inflation Factor) analysis across all 13 numeric features: every score between ~1.0 and ~1.7 — far under the standard 5-10 multicollinearity concern threshold.
- **Decision: Keep all features as-is. No drops or merges.** The independent noise deliberately injected during data generation (e.g., a skilled engineer with a weak GitHub profile) successfully kept each pair distinct enough to justify separate inclusion.
- **Why this matters for the resume narrative:** this is a complete, demonstrable example of evidence-based feature selection — proposed a concern, deferred the decision, gathered quantitative evidence (correlation + VIF), and resolved it with data rather than intuition. Stronger than either "I included everything" or "I dropped things that felt redundant."

### D19 — Business Impact Framing (Deferred)
- **Ask:** Project shouldn't just end at "model built" — it should demonstrate concrete business impact (e.g., "reduces recruiter screening effort by X%, increases productivity by Y%"), the way a real internship deliverable would be framed.
- **Key principle agreed:** Any impact numbers must be **honestly derived**, not invented. Since there's no live HR team using this system, we can't measure a true real-world effort reduction — but we can simulate the screening workflow and compute genuine, defensible proxy metrics:
  - Time/effort proxy (e.g., minutes-per-review assumption × candidates reviewed, manual vs. top-K-ranked)
  - **Recall-at-top-K%** (standard ranking evaluation: "reviewing only the top 20% ranked candidates still captures X% of true hires")
  - Time-to-shortlist comparison (qualitative automation argument)
  - Possibly a consistency/fairness-based argument, handled carefully to avoid overstating
- **Status: Deferred.** To be built as its own pipeline stage ("Business Impact Simulation") after classification + ranking are complete, so it's computed from real model outputs rather than designed first and retrofitted. Revisit after core pipeline (classification, ranking, clustering, SHAP, fairness) is done.

### D21 — Notebook Markdown Policy (For Submission)
- **Ask:** Notebooks built collaboratively are fine for code, but markdown explanations/headers in the version submitted to the instructor must be written by the user, in their own words — not phrased as referencing our chat or decision log.
- **Decision:** Going forward, notebook markdown cells contain **bullet-point notes** (what to explain, key results to reference) rather than full prose write-ups. These are drafts for the user to rewrite/expand into their own voice before submission — not submission-ready text themselves. No notebook markdown references the collaborative process, decision log, or this chat directly.
- **This decision log remains a private working reference only** — never copied into submitted notebooks.

### D22 — Classification Modeling: Results & Final Model Selection
- **Built:** `03_classification_modeling.ipynb` — baseline (Logistic Regression) → model comparison (Random Forest, XGBoost) → cross-validation → hyperparameter tuning (GridSearch on XGBoost) → probability calibration → final model saved.
- **Baseline (Logistic Regression):** Test ROC-AUC 0.753, F1 0.568, CV ROC-AUC 0.782 ± 0.027.
- **Random Forest investigation — real finding, not just a number:** at the default 0.5 threshold, Random Forest showed suspiciously low recall (0.20) despite `class_weight="balanced"`. Root cause: `class_weight` in tree-based models reweights the *split criterion* during training, but leaf-node probability outputs still reflect actual training class proportions — so the default 0.5 threshold systematically under-predicts the minority class far more than it does for Logistic Regression (where class weighting directly affects the loss function). Confirmed by inspecting the predicted-probability distribution (mean ≈ 0.28, max ≈ 0.80 — most predictions never cross 0.5). Adjusting to the F1-optimal threshold (0.335) raised F1 from 0.30 to 0.54, recall from 0.20 to 0.58. **Lesson embedded directly in the notebook:** ROC-AUC (threshold-independent) is the fairer way to compare model quality under class imbalance; F1/precision/recall at a fixed default threshold can be misleading for tree ensembles specifically.
- **XGBoost:** untuned CV ROC-AUC 0.748; after GridSearch tuning (`max_depth=3, learning_rate=0.1, n_estimators=100`), CV ROC-AUC rose to 0.783 — essentially tied with Logistic Regression (0.782).
- **Final model selection — XGBoost (tuned), not the highest single number:** Logistic Regression and tuned XGBoost are statistically near-identical on ROC-AUC. **XGBoost was still selected to carry forward** because it naturally splits on categorical interactions like `job_role` — directly relevant to validating the per-role weighting design (D10) via SHAP in a later notebook. A linear model can't represent that interaction without manually engineered terms. Documented explicitly as a deliberate trade-off, not implied as a simple "winner."
- **Calibration:** Brier score improved from 0.195 (uncalibrated) to 0.172 (calibrated) via sigmoid (Platt) scaling, confirming calibration was worth doing before using these probabilities as the Hiring Score. ROC-AUC stayed effectively stable through calibration (0.759 → 0.753) — expected, since calibration optimizes probability quality, not rank order.
- **Output:** `final_classification_model.joblib` — the calibrated, tuned XGBoost pipeline, ready for use in ranking (next notebook).

### D24 — Hiring Score & Ranking: Built and Verified
- **Built:** `04_hiring_score_and_ranking.ipynb` — loads calibrated model, generates selection probabilities, scales to 0-100 Hiring Score, ranks within `job_role` (per D3's role-local ranking design), spot-checks top/bottom candidates per role, computes recall-at-top-20% per role as a ranking-quality metric.
- **Hiring Score = probability × 100.** Deliberately just a presentation-layer rescale, not a new calculation — keeps the score traceable directly back to the calibrated model's probability output (important for explainability later).
- **Recall-at-top-20% by role:** Sales Executive 44.2%, Data Analyst 52.8%, Software Engineer 54.3%, Strategy Consultant 66.7% — all well above the ~20% a random ranking would produce, confirming the ranking concentrates true hires near the top. **This same metric is earmarked for reuse in the deferred Business Impact Simulation (D19)** — it directly supports an "X% effort reduction while still catching Y% of true hires" claim.
- **Spot-check finding:** top-ranked Sales Executive candidates show notably high `communication_score` even when `interview_score` is unremarkable (e.g., top candidate: communication 85.0, interview 51.4) — consistent with the D10 weight table's design (communication weighted "Very High" for Sales). Confirms the model is using role-appropriate signal, not just defaulting to one dominant feature.
- **Outputs:** `ranked_candidates.csv` (recruiter-facing: id, role, score, rank, percentile) and `recruitment_candidates_scored.csv` (full feature set + score/rank, for downstream notebooks — clustering, SHAP, fairness audit, Streamlit app).

### D26 — Notebook Structure: Clustering + SHAP + Fairness Audit Combined
- **Question:** Build Clustering, SHAP Explainability, and Fairness Audit as three separate notebooks, or combine into one?
- **Discussion:** These are three distinct analyses with different purposes (unsupervised segmentation; model interpretability; bias/ethics audit) and don't share a strict dependency chain — clustering doesn't need the classification model at all, while SHAP and the fairness audit both do. Separating them mirrors how Classification and Ranking were kept apart despite being related, and makes each analysis easier to point to individually when explaining the project (e.g., in an interview).
- **Decision:** **Combine into one notebook** (`05_clustering_shap_fairness.ipynb`), with clear Part A/B/C section structure (mirroring the earlier inspection+cleaning+EDA notebook) to preserve the methodological distinction even within a single file.

### D28 — Fairness Audit: No Injected Bias, Reported Honestly
- **Finding during build:** `age`, `gender`, and `relocation_preference` were generated as statistically independent draws during data simulation (D9/D13) — no deliberate correlation between sensitive attributes and outcomes exists in the dataset. Checking disparate impact (gender: all ratios ≥0.867, comfortably above the common 80% rule threshold) and age correlation with hiring_score (r=0.115, weak, explained by age's legitimate correlation with years_experience) confirms there's no meaningful bias to detect.
- **Question raised:** Should we deliberately inject bias into the dataset so the fairness audit has something concrete to "catch," making for a more dramatic result?
- **Decision: No — keep the data as generated, report the honest "no significant bias found" result.**
- **Why:** Injecting bias specifically so our own audit can detect it would test whether the audit finds what we put there, not whether the audit methodology itself is sound. The honest, complete result is: "the audit methodology (disparate impact ratios, correlation checks) is demonstrated and would catch proxy bias if present; in this dataset, none was found because sensitive attributes were generated independently of outcome by design." This is a legitimate, explainable limitation to state directly in the report, not a weakness to hide.

### D29 — SHAP Per-Role Validation: Real Findings (Including a Genuine Limitation)
- **Built and tested per-role SHAP comparison** to validate the D10 weight-table design (do roles actually show the different feature importances we designed?).
- **Finding 1 — absolute SHAP magnitudes are diluted by a known design artifact:** `interview_score` was generated in D13/Section 7 as a weighted blend of `technical_skill_score`, `aptitude_score`, and `communication_score` plus noise (correlations 0.22-0.35 with each). This means SHAP importance gets split between `interview_score` and its underlying components rather than concentrating cleanly on any one feature — `interview_score` and `project_quality_score` rank above `technical_skill_score` globally as a result. This is a real and explainable consequence of the generation design (distinct from the VIF/multicollinearity check, which only tests pairwise feature redundancy, not this three-way blending effect on attribution) — not a modeling error.
- **Finding 2 — relative (per-role-normalized) SHAP importance validates the design well:** Software Engineer shows the highest relative importance for `technical_skill_score` and `github_coding_profile_score` (both designed as "High" for SWE); Sales Executive shows highest for `communication_score` (designed "Very High"); Strategy Consultant shows highest for `education_level` (designed "Very High"). These all match D10's intended design.
- **Finding 3 — one genuine, worth-discussing mismatch:** `aptitude_score` showed *highest* relative SHAP importance for Sales Executive, not Strategy Consultant as designed ("Very High" for Consultant, "Low" for Sales). Investigated root cause: Consultant's aptitude scores cluster at a high mean (~72) with the same spread as other roles, meaning *less relative variation exists within that role's pool* for the model to use as a distinguishing signal — high average aptitude raises the role's overall baseline fit_score, but contributes less *marginal* predictive power once most candidates already have it. Sales Executive's lower mean (~55) with similar spread gives the model more discriminative range to use. **Key lesson:** SHAP importance reflects how much a feature's variation drives predictions within a subgroup — not the same thing as how heavily a feature was weighted in the generating formula, or how high its average value is for that group. A legitimate, explainable nuance, not a bug.

### D31 — Streamlit App: Two-Mode Design
- **Built:** `app.py` — two-mode Streamlit app: (1) Evaluate a Candidate (form input → Hiring Score + tier + rank/percentile + SHAP explanation chart), (2) Browse Ranked Pool (filter by role/tier/score → styled leaderboard + tier distribution chart + CSV download).
- **Key design decisions:** candidate_tier computed on-the-fly from hiring_score thresholds at load time rather than read from a pre-saved CSV column — removes dependency on the clustering notebook's output and makes the app self-contained with just the model + scored CSV. SHAP explanation uses local (per-candidate) analysis, not global importance — matches the app's actual use case (explaining a specific candidate's score, not describing the model's overall behavior). Tier thresholds: Excellent ≥50, Strong ≥35, Average ≥20, Needs Improvement <20.
- **To run locally:** `streamlit run app.py` from the project directory (requires `recruitment_candidates_scored.csv` and `final_classification_model.joblib` in the same folder).

### D32 — GitHub README
- **Built:** `README.md` — structured GitHub repository README covering: project overview with headline business impact table (52% recall, 2.6× lift, 80% effort reduction at top 20%); job roles and selection rates; full project structure map; ML pipeline summary (model comparison table, clustering, SHAP, fairness); app running instructions including the sklearn version fix; key technical decisions table (8 decisions with choice + reasoning); limitations; technologies used; and author section.
- **Placement:** root of the GitHub repository — renders automatically as the project homepage.

### D33 — GitHub Repo Description
- **Decided:** Best short description for the GitHub repository "About" field (≤350 characters):
  *"Smart Recruitment Assistant — End-to-end ML pipeline for candidate scoring, ranking, and segmentation using XGBoost, SHAP explainability, and KMeans clustering. Features a fairness/bias audit, business impact simulation (2.6× lift over random screening), and a live Streamlit app for candidate evaluation."*
- **Character count:** 298 — fits within GitHub's limit.
- **Recommended topic tags:** `machine-learning` `xgboost` `shap` `streamlit` `recruitment` `hr-analytics` `python` `data-science`

### D34 — Project Report
- **Built:** `Smart_Recruitment_Assistant_Report.docx` — 13-page Word document with professional formatting (cover page, section headings, tables, justified body text). Sections: Abstract, Introduction, Dataset Design and Generation, Methodology, Results, Streamlit Application, Limitations and Ethical Considerations, Conclusions, Technologies Used, References.
- **Key design principle:** every claim in the report is grounded in actual results from the notebooks — no invented numbers. Business impact figures cite the documented 5-min assumption explicitly. Fairness section clearly states what the audit does and does not prove. SHAP section explains the aptitude mismatch honestly rather than glossing over it.
- **Before submitting:** fill in [Your Name], [Your Institution], [Submission Date] on the cover page; rewrite notebook markdown cells in your own words (per D21).

### D35 — Sklearn Version Mismatch Fix
- **Issue encountered:** `AttributeError: module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'` when loading `final_classification_model.joblib` on the user's local machine (Python 3.14, different sklearn version than sandbox).
- **Root cause:** joblib serialises models using Python pickle, which embeds internal sklearn class names. When sklearn changes internal class names between versions (as it did with `_RemainderColsList`), loading a model saved on a different version fails.
- **Fix:** `rebuild_model.py` — a standalone script that re-runs the full training pipeline (GridSearch + calibration, ~60 seconds) on the local machine, saving a fresh `final_classification_model.joblib` built with the user's own sklearn version. Never downgrade sklearn to fix pickle compatibility — always retrain on the target machine.

### D36 — Project Documentation Practice
- **Decision:** Maintain this living decision log throughout the project, capturing: the question/problem, options considered, final decision + reasoning, challenges/pushback, and status. Updated on request or proactively after major decisions.
- **Why:** Doubles as (a) interview prep material — concrete answers to "tell me about a decision you made and why" — and (b) raw material for the final project report's methodology/limitations sections.

---

## Open / Pending Items
- [x] Finalize data simulation logic (feature distributions, role-conditional rules, label-generation formula) — weight table + selection rates locked (D9-D11); generation code next
- [x] Generate dataset — `01_data_generation.ipynb` built, executed end-to-end, verified (D13). Output: `recruitment_candidates.csv`
- [x] Run correlation/VIF check → resolve D7 redundancy questions with evidence — resolved in D17, no features dropped
- [x] Data inspection, cleaning, EDA — `02_inspection_cleaning_eda.ipynb` built, executed, verified (D15, D16, D17). Output: `recruitment_candidates_cleaned.csv`
- [x] Classification modeling (baseline → comparison → tuning → calibration) — `03_classification_modeling.ipynb` built, executed, verified (D22). Final model: calibrated, tuned XGBoost. Output: `final_classification_model.joblib`
- [x] Hiring Score + per-role ranking implementation — `04_hiring_score_and_ranking.ipynb` built, executed, verified (D24). Outputs: `ranked_candidates.csv`, `recruitment_candidates_scored.csv`
- [x] Clustering (KMeans vs Hierarchical vs GMM, optimal k, PCA viz, cluster labeling) — combined into `05_clustering_shap_fairness.ipynb` (D26, D29)
- [x] SHAP explainability (global + per-role) — combined into `05_clustering_shap_fairness.ipynb`, validated D10 weight-table design with one investigated mismatch (D29)
- [x] Fairness/bias audit (gender/age proxy bias check) — combined into `05_clustering_shap_fairness.ipynb`, no bias found, reported honestly with limitation stated (D28)
- [x] Streamlit app — `app.py` built and verified (D31)
- [x] Business impact simulation (deferred, D19) — `06_business_impact_simulation.ipynb` built, executed, verified. Headline: top 20% shortlist captures 52% of true hires (2.6x lift over random), 80% effort reduction, ~267hrs estimated time saved (documented 5-min assumption).
- [x] Final report + GitHub README — `Smart_Recruitment_Assistant_Report.docx` (13 pages, D34) and `README.md` (D32) built. GitHub repo description finalised (D33). Sklearn version fix documented as `rebuild_model.py` (D35).

---

## PROJECT COMPLETE

All pipeline stages delivered and verified:

| Stage | Notebook / File | Key Output |
|---|---|---|
| Data Generation | `01_data_generation.ipynb` | `recruitment_candidates.csv` |
| Inspection + Cleaning + EDA | `02_inspection_cleaning_eda.ipynb` | `recruitment_candidates_cleaned.csv` |
| Classification Modeling | `03_classification_modeling.ipynb` | `final_classification_model.joblib` |
| Hiring Score + Ranking | `04_hiring_score_and_ranking.ipynb` | `ranked_candidates.csv`, `recruitment_candidates_scored.csv` |
| Clustering + SHAP + Fairness | `05_clustering_shap_fairness.ipynb` | Tier labels, SHAP validation, fairness audit |
| Business Impact Simulation | `06_business_impact_simulation.ipynb` | 2.6× lift, 52% recall, 80% effort reduction |
| Streamlit App | `app.py` | Live candidate evaluation + ranked pool browser |
| Model Rebuild Script | `rebuild_model.py` | Fixes sklearn version mismatch |
| GitHub README | `README.md` | Project homepage |
| Project Report | `Smart_Recruitment_Assistant_Report.docx` | 13-page submission document |
| Decision Log | `decision_log.md` | Full reasoning trail (D1–D35) |
