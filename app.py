import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean, professional palette */
    :root {
        --primary: #1a365d;
        --accent: #3182ce;
        --success: #276749;
        --warning: #744210;
        --danger: #742a2a;
        --surface: #f7fafc;
        --border: #e2e8f0;
    }

    .main { background-color: #f7fafc; }

    /* Score card */
    .score-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 4px solid #3182ce;
        margin-bottom: 16px;
    }
    .score-number {
        font-size: 64px;
        font-weight: 700;
        color: #1a365d;
        line-height: 1;
    }
    .score-label {
        font-size: 13px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    /* Tier badges */
    .tier-excellent  { background:#c6f6d5; color:#276749; padding:6px 16px; border-radius:20px; font-weight:600; display:inline-block; }
    .tier-strong     { background:#bee3f8; color:#2b6cb0; padding:6px 16px; border-radius:20px; font-weight:600; display:inline-block; }
    .tier-average    { background:#fefcbf; color:#744210; padding:6px 16px; border-radius:20px; font-weight:600; display:inline-block; }
    .tier-needs      { background:#fed7d7; color:#742a2a; padding:6px 16px; border-radius:20px; font-weight:600; display:inline-block; }

    /* Section headers */
    .section-header {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #718096;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Rank row */
    .rank-row {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load assets (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("final_classification_model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("recruitment_candidates_scored.csv")
    # Compute candidate_tier on load (avoids dependency on clustering notebook's saved output)
    def get_tier_local(score):
        if score >= 50:   return "Excellent"
        if score >= 35:   return "Strong"
        if score >= 20:   return "Average"
        return "Needs Improvement"
    df["candidate_tier"] = df["hiring_score"].apply(get_tier_local)
    return df

@st.cache_resource
def get_shap_explainer(_model):
    base_pipeline = _model.calibrated_classifiers_[0].estimator
    xgb_model = base_pipeline.named_steps["clf"]
    return shap.TreeExplainer(xgb_model)

model = load_model()
df = load_data()
explainer = get_shap_explainer(model)

base_pipeline = model.calibrated_classifiers_[0].estimator
preprocessor = base_pipeline.named_steps["pre"]
feature_names = list(preprocessor.get_feature_names_out())

FEATURE_COLS = ["job_role", "education_level", "years_experience", "technical_skill_score",
    "aptitude_score", "communication_score", "interview_score", "internship_experience",
    "projects_count", "project_quality_score", "certifications_count", "certification_prestige_score",
    "competition_awards_count", "ats_score", "linkedin_profile_score", "github_coding_profile_score",
    "relocation_preference"]

TIER_ORDER = ["Excellent", "Strong", "Average", "Needs Improvement"]
TIER_COLORS = {"Excellent": "#276749", "Strong": "#2b6cb0", "Average": "#744210", "Needs Improvement": "#742a2a"}

def get_tier(score):
    if score >= 50:   return "Excellent"
    if score >= 35:   return "Strong"
    if score >= 20:   return "Average"
    return "Needs Improvement"

def score_candidate(inputs: dict):
    input_df = pd.DataFrame([inputs])
    proba = model.predict_proba(input_df)[0, 1]
    hiring_score = round(proba * 100, 1)
    return hiring_score

def get_shap_explanation(inputs: dict, top_n=6):
    input_df = pd.DataFrame([inputs])
    X_transformed = preprocessor.transform(input_df)
    shap_vals = explainer.shap_values(X_transformed)[0]
    clean_names = [
        n.replace("remainder__", "").replace("cat__", "").replace("_", " ").title()
        for n in feature_names
    ]
    shap_df = pd.DataFrame({"Feature": clean_names, "SHAP": shap_vals})
    shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index)
    return shap_df.head(top_n).reset_index(drop=True)

def get_rank(hiring_score, role):
    role_scores = df[df["job_role"] == role]["hiring_score"]
    rank = int((role_scores > hiring_score).sum()) + 1
    total = len(role_scores)
    percentile = round((1 - (rank - 1) / total) * 100, 1)
    return rank, total, percentile

# ── Sidebar navigation ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Smart Recruitment\nAssistant")
    st.markdown("---")
    mode = st.radio(
        "Choose a task",
        ["Evaluate a Candidate", "Browse Ranked Pool"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#718096;'>"
        "Model: XGBoost (calibrated)<br>"
        "Dataset: 4,000 candidates<br>"
        "Roles: 4 job tracks<br>"
        "Version: 1.0"
        "</div>",
        unsafe_allow_html=True
    )

# ── MODE 1: Evaluate a candidate ──────────────────────────────
if mode == "Evaluate a Candidate":
    st.markdown("## Evaluate a Candidate")
    st.markdown("Enter the candidate's details below to generate a Hiring Score, tier, and explanation.")
    st.markdown("---")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="section-header">Role & Background</div>', unsafe_allow_html=True)

        job_role = st.selectbox("Job Role", ["Software Engineer", "Data Analyst", "Sales Executive", "Strategy Consultant"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        relocation_preference = st.selectbox("Relocation Preference", ["Yes", "No", "Open to Remote"])
        years_experience = st.slider("Years of Experience", 0.0, 15.0, 3.0, 0.5)
        internship_experience = st.selectbox("Had Internship(s)?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown('<div class="section-header" style="margin-top:20px;">Assessment Scores</div>', unsafe_allow_html=True)
        technical_skill_score = st.slider("Technical Skill Score", 0.0, 100.0, 65.0, 1.0)
        aptitude_score = st.slider("Aptitude Score", 0.0, 100.0, 65.0, 1.0)
        communication_score = st.slider("Communication Score", 0.0, 100.0, 65.0, 1.0)
        interview_score = st.slider("Interview Score", 0.0, 100.0, 65.0, 1.0)

    with col_right:
        st.markdown('<div class="section-header">Projects & Certifications</div>', unsafe_allow_html=True)
        projects_count = st.slider("Number of Projects", 0, 12, 3)
        project_quality_score = st.slider("Project Quality Score", 0.0, 100.0, 55.0, 1.0)
        certifications_count = st.slider("Number of Certifications", 0, 8, 1)
        certification_prestige_score = st.slider("Certification Prestige Score", 0.0, 100.0, 50.0, 1.0)
        competition_awards_count = st.slider("Competition / Hackathon Awards", 0, 5, 0)

        st.markdown('<div class="section-header" style="margin-top:20px;">Digital Presence</div>', unsafe_allow_html=True)
        ats_score = st.slider("ATS Resume Match Score", 0.0, 100.0, 55.0, 1.0)
        linkedin_profile_score = st.slider("LinkedIn Profile Score", 0.0, 10.0, 5.5, 0.5)
        github_coding_profile_score = st.slider("GitHub / Coding Profile Score", 0.0, 10.0, 5.0, 0.5)

    st.markdown("---")
    evaluate_clicked = st.button("Evaluate Candidate", type="primary", use_container_width=True)

    if evaluate_clicked:
        candidate_inputs = {
            "job_role": job_role, "education_level": education_level,
            "years_experience": years_experience, "technical_skill_score": technical_skill_score,
            "aptitude_score": aptitude_score, "communication_score": communication_score,
            "interview_score": interview_score, "internship_experience": internship_experience,
            "projects_count": projects_count, "project_quality_score": project_quality_score,
            "certifications_count": certifications_count, "certification_prestige_score": certification_prestige_score,
            "competition_awards_count": competition_awards_count, "ats_score": ats_score,
            "linkedin_profile_score": linkedin_profile_score, "github_coding_profile_score": github_coding_profile_score,
            "relocation_preference": relocation_preference,
        }

        hiring_score = score_candidate(candidate_inputs)
        tier = get_tier(hiring_score)
        rank, total, percentile = get_rank(hiring_score, job_role)
        shap_df = get_shap_explanation(candidate_inputs)

        st.markdown("---")
        st.markdown("### Results")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number">{hiring_score}</div>'
                f'<div class="score-label">Hiring Score</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r2:
            tier_class = {"Excellent": "tier-excellent", "Strong": "tier-strong", "Average": "tier-average", "Needs Improvement": "tier-needs"}[tier]
            st.markdown(
                f'<div class="score-card">'
                f'<div style="margin-top:10px;"><span class="{tier_class}">{tier}</span></div>'
                f'<div class="score-label" style="margin-top:12px;">Candidate Tier</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r3:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" style="font-size:48px;">#{rank}</div>'
                f'<div class="score-label">of {total} {job_role}s</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r4:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" style="font-size:48px;">Top {percentile}%</div>'
                f'<div class="score-label">Percentile in role</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # SHAP explanation
        st.markdown("### Why this score?")
        st.caption("The chart below shows the top factors driving this candidate's score. Positive values pushed the score up; negative values pulled it down.")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#276749" if v > 0 else "#742a2a" for v in shap_df["SHAP"]]
        bars = ax.barh(shap_df["Feature"][::-1], shap_df["SHAP"][::-1], color=colors[::-1], height=0.6)
        ax.axvline(0, color="#718096", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on hiring score prediction)")
        ax.set_title("Top contributing factors", fontsize=12, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── MODE 2: Browse ranked pool ─────────────────────────────────
else:
    st.markdown("## Ranked Candidate Pool")
    st.markdown("Browse and filter all evaluated candidates, ranked within their job role.")
    st.markdown("---")

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_role = st.selectbox("Filter by Job Role", ["All roles"] + sorted(df["job_role"].unique().tolist()))
    with f2:
        selected_tiers = st.multiselect("Filter by Tier", TIER_ORDER, default=TIER_ORDER)
    with f3:
        min_score = st.slider("Minimum Hiring Score", 0, 80, 0)

    filtered = df.copy()
    if selected_role != "All roles":
        filtered = filtered[filtered["job_role"] == selected_role]
    if selected_tiers:
        filtered = filtered[filtered["candidate_tier"].isin(selected_tiers)]
    filtered = filtered[filtered["hiring_score"] >= min_score]
    filtered = filtered.sort_values(["job_role", "rank_in_role"])

    st.markdown(f"**{len(filtered):,} candidates** match your filters")

    display_cols = ["candidate_id", "job_role", "candidate_tier", "hiring_score",
                    "rank_in_role", "percentile_in_role", "selected"]
    renamed = {
        "candidate_id": "ID", "job_role": "Role", "candidate_tier": "Tier",
        "hiring_score": "Score", "rank_in_role": "Rank", "percentile_in_role": "Percentile",
        "selected": "Actual Outcome"
    }

    display_df = filtered[display_cols].rename(columns=renamed).reset_index(drop=True)
    display_df["Actual Outcome"] = display_df["Actual Outcome"].map({1: "✅ Selected", 0: "❌ Rejected"})

    def color_tier(val):
        colors = {
            "Excellent": "background-color:#c6f6d5; color:#276749",
            "Strong": "background-color:#bee3f8; color:#2b6cb0",
            "Average": "background-color:#fefcbf; color:#744210",
            "Needs Improvement": "background-color:#fed7d7; color:#742a2a",
        }
        return colors.get(val, "")

    styled = display_df.style.applymap(color_tier, subset=["Tier"])
    st.dataframe(styled, use_container_width=True, height=500)

    csv_data = filtered[display_cols].rename(columns=renamed).to_csv(index=False)
    st.download_button(
        "Download filtered results as CSV",
        data=csv_data,
        file_name="filtered_candidates.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Summary stats below the table
    st.markdown("---")
    st.markdown("### Pool Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Hiring Score", f"{filtered['hiring_score'].mean():.1f}")
    s2.metric("Overall Selection Rate", f"{filtered['selected'].mean():.1%}")
    s3.metric("Excellent / Strong", f"{(filtered['candidate_tier'].isin(['Excellent','Strong'])).sum():,}")
    s4.metric("Roles in view", filtered["job_role"].nunique())

    # Tier distribution chart
    tier_counts = filtered["candidate_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bar_colors = [TIER_COLORS[t] for t in TIER_ORDER]
    ax2.bar(TIER_ORDER, tier_counts.values, color=bar_colors, width=0.55)
    ax2.set_ylabel("Number of candidates")
    ax2.set_title("Tier distribution", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    for i, (v, t) in enumerate(zip(tier_counts.values, TIER_ORDER)):
        ax2.text(i, v + 3, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
