"""
app.py — OmniHealth AI · Smart Healthcare Analytics Suite
==========================================================
Solo project by: [Your Name]
Run with:  streamlit run app.py
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "patients.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "metadata.pkl")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniHealth AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #091525 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(13,27,46,0.8);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.2);
}
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }

/* Section headers */
h1 { color: #f0f9ff !important; font-weight: 700; }
h2 { color: #bae6fd !important; font-weight: 600; }
h3 { color: #7dd3fc !important; font-weight: 500; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.6rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(14,165,233,0.5);
}

/* Input widgets */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-weight: 500;
    font-size: 0.85rem;
}

/* DataFrames */
.dataframe { border-radius: 10px; overflow: hidden; }

/* Risk badge helper classes (used in HTML) */
.badge-high   { background:#ef4444; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#f59e0b; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#22c55e; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(14,165,233,0.15) 0%, rgba(99,102,241,0.15) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 6px; }

/* Prediction result box */
.pred-box {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-top: 12px;
    font-size: 1.1rem;
    font-weight: 600;
}
.pred-high { background: rgba(239,68,68,0.15); border: 1.5px solid #ef4444; color:#fca5a5; }
.pred-low  { background: rgba(34,197,94,0.15);  border: 1.5px solid #22c55e; color:#86efac; }

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 20px 0;
}

/* Anomaly row highlight */
.anomaly-tag { color:#f87171; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — load resources (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_models():
    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta   = joblib.load(META_PATH)
    return clf, scaler, meta


def models_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def data_ready() -> bool:
    return os.path.exists(DATA_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Plotly theme helper
# ══════════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,46,0.6)",
    font_color="#94a3b8",
    font_family="Inter",
    margin=dict(t=40, b=20, l=20, r=20),
    title_font_color="#bae6fd",
)
PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#f472b6", "#fb923c", "#facc15"]


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size:2.4rem;'>🏥</span>
        <h2 style='color:#38bdf8; margin:0; font-size:1.3rem; font-weight:700;'>OmniHealth AI</h2>
        <p style='color:#64748b; font-size:0.75rem; margin:2px 0 0 0;'>Smart Healthcare Analytics</p>
    </div>
    <div style='height:1px; background:rgba(99,179,237,0.2); margin:12px 0 20px 0;'></div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠  Dashboard",
            "🩺  Risk Predictor",
            "📋  Patient Table",
            "🚨  Anomaly Detector",
            "📁  Upload Your CSV",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div style='height:1px; background:rgba(99,179,237,0.2); margin:20px 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='color:#64748b; font-size:0.72rem; padding: 0 4px;'>
        <b style='color:#475569;'>Dataset Features</b><br>
        Age · Sex · Chest Pain · Blood Pressure<br>
        Cholesterol · Max Heart Rate · Angina<br>
        ST Depression · Vessels · Thalassemia
    </div>
    """, unsafe_allow_html=True)

    return page.split("  ")[1]   # strip icon prefix


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard(df: pd.DataFrame, meta: dict):
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>🏥 OmniHealth AI Dashboard</p>
        <p class='hero-sub'>AI-Powered Smart Healthcare Analytics · Real-time Patient Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    total     = len(df)
    diseased  = int(df["target"].sum())
    healthy   = total - diseased
    avg_age   = round(df["age"].mean(), 1)
    avg_chol  = round(df["chol"].mean(), 1)
    acc       = meta.get("accuracy", "—")
    high_risk_pct = round(diseased / total * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Patients", f"{total:,}")
    c2.metric("❤️ Disease Cases", f"{diseased:,}", f"{high_risk_pct}% of total")
    c3.metric("✅ Healthy Patients", f"{healthy:,}")
    c4.metric("🎂 Average Age", f"{avg_age} yrs")
    c5.metric("🤖 Model Accuracy", f"{acc}%")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Row 1: Pie + Age hist ─────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🫀 Disease Distribution")
        pie = px.pie(
            names=["Heart Disease", "No Disease"],
            values=[diseased, healthy],
            color_discrete_sequence=["#ef4444", "#22c55e"],
            hole=0.55,
        )
        pie.update_layout(**PLOT_LAYOUT)
        pie.update_traces(textfont_size=13, pull=[0.04, 0])
        st.plotly_chart(pie, use_container_width=True)

    with col_b:
        st.markdown("### 📊 Age Distribution")
        hist = px.histogram(
            df, x="age", nbins=25, color_discrete_sequence=["#38bdf8"],
        )
        hist.update_layout(**PLOT_LAYOUT, bargap=0.05)
        hist.update_traces(marker_line_color="#0ea5e9", marker_line_width=1)
        st.plotly_chart(hist, use_container_width=True)

    # ── Row 2: Cholesterol box + Sex split ───────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("### 🩸 Cholesterol by Disease Status")
        df_c = df.copy()
        df_c["Status"] = df_c["target"].map({1: "Disease", 0: "No Disease"})
        box = px.box(
            df_c, x="Status", y="chol",
            color="Status",
            color_discrete_map={"Disease": "#f87171", "No Disease": "#4ade80"},
        )
        box.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(box, use_container_width=True)

    with col_d:
        st.markdown("### 🚻 Disease Rate by Sex")
        sex_df = df.groupby(["sex", "target"]).size().reset_index(name="count")
        sex_df["Sex"]    = sex_df["sex"].map({1: "Male", 0: "Female"})
        sex_df["Status"] = sex_df["target"].map({1: "Disease", 0: "No Disease"})
        bar = px.bar(
            sex_df, x="Sex", y="count", color="Status",
            barmode="group",
            color_discrete_map={"Disease": "#f87171", "No Disease": "#4ade80"},
        )
        bar.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(bar, use_container_width=True)

    # ── Row 3: Feature importance ─────────────────────────────────────────────
    st.markdown("### 🧠 AI Feature Importance (Which Symptoms Matter Most?)")
    fi = meta.get("feature_importance", {})
    fi_df = pd.DataFrame(
        sorted(fi.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importance"]
    )
    fi_df["Importance%"] = (fi_df["Importance"] * 100).round(1)
    bar2 = px.bar(
        fi_df, x="Importance%", y="Feature",
        orientation="h",
        color="Importance%",
        color_continuous_scale=["#1e3a5f", "#38bdf8", "#818cf8"],
    )
    bar2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(bar2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Risk Predictor
# ══════════════════════════════════════════════════════════════════════════════

def page_risk_predictor(clf, scaler):
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>🩺 Patient Risk Predictor</p>
        <p class='hero-sub'>Enter patient symptoms below — AI will predict heart disease risk instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        st.markdown("#### 📝 Patient Symptom Input")
        with st.form("risk_form"):
            r1c1, r1c2 = st.columns(2)
            age    = r1c1.slider("Age",            20, 80, 50)
            sex    = r1c2.selectbox("Sex",          ["Male (1)", "Female (0)"])
            sex_v  = 1 if "Male" in sex else 0

            r2c1, r2c2 = st.columns(2)
            cp_map = {
                "Typical Angina (0)": 0,
                "Atypical Angina (1)": 1,
                "Non-Anginal Pain (2)": 2,
                "Asymptomatic (3)": 3,
            }
            cp     = r2c1.selectbox("Chest Pain Type", list(cp_map.keys()))
            cp_v   = cp_map[cp]
            trestbps = r2c2.slider("Resting Blood Pressure (mmHg)", 90, 210, 130)

            r3c1, r3c2 = st.columns(2)
            chol   = r3c1.slider("Cholesterol (mg/dl)", 100, 600, 240)
            thalach= r3c2.slider("Max Heart Rate (bpm)", 60, 210, 150)

            r4c1, r4c2 = st.columns(2)
            fbs    = r4c1.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])
            fbs_v  = 1 if "Yes" in fbs else 0
            exang  = r4c2.selectbox("Exercise Induced Angina?", ["No (0)", "Yes (1)"])
            exang_v= 1 if "Yes" in exang else 0

            r5c1, r5c2 = st.columns(2)
            oldpeak= r5c1.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
            ca     = r5c2.slider("Major Vessels (0–3)", 0, 3, 0)

            r6c1, r6c2 = st.columns(2)
            restecg_map = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
            restecg = r6c1.selectbox("Resting ECG", list(restecg_map.keys()))
            restecg_v = restecg_map[restecg]
            slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
            slope   = r6c2.selectbox("ST Slope", list(slope_map.keys()))
            slope_v = slope_map[slope]

            thal_map = {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}
            thal = st.selectbox("Thalassemia", list(thal_map.keys()))
            thal_v = thal_map[thal]

            submitted = st.form_submit_button("🔍 Run AI Prediction", use_container_width=True)

    with col_result:
        st.markdown("#### 🤖 AI Prediction Result")
        if submitted:
            features = np.array([[
                age, sex_v, cp_v, trestbps, chol, fbs_v,
                restecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v
            ]], dtype=float)
            features_scaled = scaler.transform(features)
            prediction = clf.predict(features_scaled)[0]
            probability = clf.predict_proba(features_scaled)[0]
            risk_pct = round(probability[1] * 100, 1)

            if prediction == 1:
                st.markdown(f"""
                <div class='pred-box pred-high'>
                    ❤️‍🔥 HIGH RISK DETECTED<br>
                    <span style='font-size:2.4rem; font-weight:800;'>{risk_pct}%</span><br>
                    <span style='font-size:0.85rem; font-weight:400;'>probability of Heart Disease</span>
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ **Recommendation:** Immediate cardiology consultation advised. Lifestyle changes and medication review recommended.")
            else:
                st.markdown(f"""
                <div class='pred-box pred-low'>
                    💚 LOW RISK<br>
                    <span style='font-size:2.4rem; font-weight:800;'>{risk_pct}%</span><br>
                    <span style='font-size:0.85rem; font-weight:400;'>probability of Heart Disease</span>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ **Recommendation:** Patient appears healthy. Continue regular annual check-ups.")

            # Probability gauge
            st.markdown("<br>", unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risk Score %", "font": {"color": "#94a3b8", "size": 14}},
                number={"suffix": "%", "font": {"color": "#38bdf8", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#ef4444" if prediction == 1 else "#22c55e"},
                    "steps": [
                        {"range": [0,  40], "color": "rgba(34,197,94,0.15)"},
                        {"range": [40, 70], "color": "rgba(234,179,8,0.15)"},
                        {"range": [70,100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.75,
                        "value": 60,
                    },
                    "bgcolor": "rgba(13,27,46,0.8)",
                    "bordercolor": "rgba(99,179,237,0.2)",
                },
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=240,
                margin=dict(t=20, b=10, l=20, r=20),
            )
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("👈 Fill in the patient details and click **Run AI Prediction** to see results.")
            st.markdown("""
            <div style='margin-top:16px; color:#475569; font-size:0.82rem; line-height:1.7;'>
            <b style='color:#64748b;'>How it works:</b><br>
            1. Our Random Forest model was trained on 400 patient records<br>
            2. It analyses 13 clinical features simultaneously<br>
            3. Returns a disease probability percentage<br>
            4. Accuracy: see Dashboard for model metrics
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Patient Table
# ══════════════════════════════════════════════════════════════════════════════

def page_patient_table(df: pd.DataFrame, clf, scaler):
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>📋 Patient Risk Table</p>
        <p class='hero-sub'>All patients ranked by AI-predicted heart disease risk score.</p>
    </div>
    """, unsafe_allow_html=True)

    # Predict probabilities for all patients
    X = df[FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = clf.predict(X_scaled)

    display_df = df.copy()
    display_df["Risk %"]      = (probs * 100).round(1)
    display_df["Prediction"]  = preds
    display_df["Risk Level"]  = display_df["Risk %"].apply(
        lambda v: "🔴 High" if v >= 65 else ("🟡 Medium" if v >= 40 else "🟢 Low")
    )
    display_df["Actual"]      = display_df["target"].map({1: "✅ Disease", 0: "💚 Healthy"})

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    risk_filter  = col_f1.selectbox("Filter by Risk Level", ["All", "🔴 High", "🟡 Medium", "🟢 Low"])
    min_age, max_age = col_f2.slider("Age Range", 20, 80, (20, 80))
    search = col_f3.text_input("🔍 Search by Name")

    filtered = display_df.copy()
    filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]
    if search.strip():
        if "name" in filtered.columns:
            filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered)}** patients shown")

    show_cols = ["name", "age", "sex", "chol", "trestbps", "thalach", "Risk %", "Risk Level", "Actual"] \
        if "name" in display_df.columns else \
        ["age", "sex", "chol", "trestbps", "thalach", "Risk %", "Risk Level", "Actual"]

    st.dataframe(
        filtered[show_cols].sort_values("Risk %", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    # Download
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Table as CSV",
        data=csv_bytes,
        file_name="omnihealth_risk_table.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════

def page_anomaly(df: pd.DataFrame, scaler):
    from sklearn.ensemble import IsolationForest

    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>🚨 Anomaly Detector</p>
        <p class='hero-sub'>Isolation Forest AI auto-detects patients with statistically abnormal clinical profiles.</p>
    </div>
    """, unsafe_allow_html=True)

    sensitivity = st.slider(
        "Detection Sensitivity (contamination %)",
        min_value=3, max_value=20, value=8, step=1,
        help="Higher % = more patients flagged as anomalous"
    )
    if st.button("🔍 Run Anomaly Detection"):
        X = df[FEATURE_COLS].values.astype(float)
        X_scaled = scaler.transform(X)

        iso = IsolationForest(
            n_estimators=150,
            contamination=sensitivity / 100,
            random_state=42,
        )
        labels = iso.fit_predict(X_scaled)   # -1 = anomaly
        scores = iso.decision_function(X_scaled)

        result_df = df.copy()
        result_df["Anomaly Score"] = scores.round(4)
        result_df["Status"]        = ["🚨 ANOMALY" if l == -1 else "✅ Normal" for l in labels]
        result_df["Actual"]        = result_df["target"].map({1: "Disease", 0: "Healthy"})

        anomalies  = result_df[result_df["Status"] == "🚨 ANOMALY"]
        normal_pts = result_df[result_df["Status"] == "✅ Normal"]

        a1, a2, a3 = st.columns(3)
        a1.metric("🚨 Anomalies Found", len(anomalies))
        a2.metric("✅ Normal Patients", len(normal_pts))
        a3.metric("📊 Anomaly Rate", f"{len(anomalies)/len(result_df)*100:.1f}%")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        col_sc, col_tbl = st.columns([1, 1.2])

        with col_sc:
            st.markdown("### 📈 Anomaly Score Distribution")
            fig = px.histogram(
                result_df, x="Anomaly Score", color="Status",
                nbins=30,
                color_discrete_map={"🚨 ANOMALY": "#ef4444", "✅ Normal": "#22c55e"},
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_tbl:
            st.markdown("### 🚨 Flagged Anomalous Patients")
            if "name" in anomalies.columns:
                show = ["name", "age", "chol", "trestbps", "thalach", "Anomaly Score", "Actual"]
            else:
                show = ["age", "chol", "trestbps", "thalach", "Anomaly Score", "Actual"]
            st.dataframe(
                anomalies[show].sort_values("Anomaly Score").reset_index(drop=True),
                use_container_width=True, height=360,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Upload CSV
# ══════════════════════════════════════════════════════════════════════════════

def page_upload(clf, scaler):
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>📁 Upload Your Own Patient CSV</p>
        <p class='hero-sub'>Upload any compatible CSV — the AI model will score every patient instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > **Required columns:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`,
    > `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
    > *(a `target` column is optional)*
    """)

    uploaded = st.file_uploader("Drop your CSV file here", type=["csv"])

    if uploaded:
        try:
            custom_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(custom_df)} rows and {custom_df.shape[1]} columns")
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(custom_df.head(), use_container_width=True)

            missing = [c for c in FEATURE_COLS if c not in custom_df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
            else:
                if st.button("🤖 Run AI Risk Scoring on Uploaded Data"):
                    X = custom_df[FEATURE_COLS].fillna(custom_df[FEATURE_COLS].median()).values.astype(float)
                    X_scaled = scaler.transform(X)
                    probs = clf.predict_proba(X_scaled)[:, 1]
                    preds = clf.predict(X_scaled)

                    out_df = custom_df.copy()
                    out_df["Risk %"]     = (probs * 100).round(1)
                    out_df["Prediction"] = preds
                    out_df["Risk Level"] = out_df["Risk %"].apply(
                        lambda v: "🔴 High" if v >= 65 else ("🟡 Medium" if v >= 40 else "🟢 Low")
                    )

                    high = (out_df["Risk Level"] == "🔴 High").sum()
                    med  = (out_df["Risk Level"] == "🟡 Medium").sum()
                    low  = (out_df["Risk Level"] == "🟢 Low").sum()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("🔴 High Risk",   high)
                    m2.metric("🟡 Medium Risk", med)
                    m3.metric("🟢 Low Risk",    low)

                    st.markdown("**Scored Patient Table:**")
                    st.dataframe(out_df.sort_values("Risk %", ascending=False), use_container_width=True, height=400)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Scored Results",
                        data=csv_bytes,
                        file_name="scored_patients.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("**No data? Download our sample dataset to see the expected format:**")
        if data_ready():
            with open(DATA_PATH, "rb") as f:
                st.download_button("📥 Download Sample Dataset", f, "sample_patients.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Setup warning (if model not trained yet)
# ══════════════════════════════════════════════════════════════════════════════

def show_setup_required():
    st.markdown("""
    <div class='hero-banner' style='border-color:rgba(239,68,68,0.4);'>
        <p class='hero-title' style='background:linear-gradient(90deg,#f87171,#fbbf24);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            ⚙️ First-Time Setup Required
        </p>
        <p class='hero-sub'>Run the two commands below to generate data and train the AI model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 3-Step Setup (run these in your terminal once):")
    st.code("""
# Step 1 — Install Python libraries
pip install -r requirements.txt

# Step 2 — Generate 500 synthetic patient records
python data/generate_data.py

# Step 3 — Train the AI model
python train_model.py

# Step 4 — Launch the dashboard
streamlit run app.py
    """, language="bash")

    st.info("After running these commands, **refresh this page** and the full dashboard will appear!")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if not data_ready() or not models_ready():
        show_setup_required()
        return

    df          = load_dataframe(DATA_PATH)
    clf, scaler, meta = load_models()

    if page == "Dashboard":
        page_dashboard(df, meta)
    elif page == "Risk Predictor":
        page_risk_predictor(clf, scaler)
    elif page == "Patient Table":
        page_patient_table(df, clf, scaler)
    elif page == "Anomaly Detector":
        page_anomaly(df, scaler)
    elif page == "Upload Your CSV":
        page_upload(clf, scaler)


if __name__ == "__main__":
    main()
