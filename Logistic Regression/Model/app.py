import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard · Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --card:      #16161f;
    --border:    #222230;
    --accent:    #7c5cfc;
    --accent2:   #e040fb;
    --safe:      #00e5a0;
    --danger:    #ff4560;
    --muted:     #6b6b88;
    --text:      #e8e8f4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ─── Hero header ─── */
.hero {
    background: linear-gradient(135deg, #13131e 0%, #1a1030 60%, #0d1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(124,92,252,.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff 30%, #a78bfa 70%, #e040fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    color: var(--muted); font-size: .95rem; margin-top: .5rem; font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(124,92,252,.15);
    border: 1px solid rgba(124,92,252,.35);
    color: #a78bfa; font-size: .75rem; font-weight: 600;
    padding: .2rem .7rem; border-radius: 99px; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: 1rem;
}

/* ─── Cards ─── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.accent::after  { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.metric-card.safe::after    { background: var(--safe); }
.metric-card.danger::after  { background: var(--danger); }

.metric-label {
    color: var(--muted); font-size: .78rem;
    font-weight: 500; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: .4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.metric-value.safe   { color: var(--safe); }
.metric-value.danger { color: var(--danger); }
.metric-value.accent { color: var(--accent); }

/* ─── Result banner ─── */
.result-banner {
    border-radius: 16px; padding: 2rem 2.4rem;
    display: flex; align-items: center; gap: 1.5rem;
    margin-top: 1.5rem;
}
.result-banner.churn {
    background: linear-gradient(135deg, rgba(255,69,96,.12), rgba(255,69,96,.04));
    border: 1px solid rgba(255,69,96,.35);
}
.result-banner.safe {
    background: linear-gradient(135deg, rgba(0,229,160,.12), rgba(0,229,160,.04));
    border: 1px solid rgba(0,229,160,.35);
}
.result-icon { font-size: 3.5rem; line-height: 1; }
.result-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 700; margin-bottom: .25rem;
}
.result-heading.churn { color: var(--danger); }
.result-heading.safe  { color: var(--safe); }
.result-sub { color: var(--muted); font-size: .9rem; }

/* ─── Section title ─── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: var(--text); margin-bottom: 1rem;
    display: flex; align-items: center; gap: .6rem;
}
.section-title span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); display: inline-block;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--muted) !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
}
.sidebar-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.sidebar-section-title {
    font-family: 'Syne', sans-serif;
    font-size: .8rem; font-weight: 700;
    color: var(--accent); letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: .75rem;
}

/* ─── Predict button ─── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: .8rem 0 !important; width: 100% !important;
    letter-spacing: .04em !important; cursor: pointer !important;
    transition: opacity .2s !important;
    box-shadow: 0 4px 24px rgba(124,92,252,.4) !important;
}
div[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ─── Progress bar override ─── */
[data-testid="stProgress"] > div > div { border-radius: 99px; }

/* ─── Divider ─── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "logistic.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "customer_churn_prediction_dataset.csv")

model = load_model()
df    = load_data()

FEATURE_COLS = list(model.feature_names_in_)
# Categorical options
CAT_OPTS = {
    "gender":            ["Male", "Female"],
    "Partner":           ["Yes", "No"],
    "Dependents":        ["Yes", "No"],
    "PhoneService":      ["Yes", "No"],
    "MultipleLines":     ["Yes", "No", "No phone service"],
    "InternetService":   ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":    ["Yes", "No", "No internet service"],
    "OnlineBackup":      ["Yes", "No", "No internet service"],
    "DeviceProtection":  ["Yes", "No", "No internet service"],
    "TechSupport":       ["Yes", "No", "No internet service"],
    "StreamingTV":       ["Yes", "No", "No internet service"],
    "StreamingMovies":   ["Yes", "No", "No internet service"],
    "Contract":          ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling":  ["Yes", "No"],
    "PaymentMethod":     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
}

def build_input(inputs: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURE_COLS}
    row["SeniorCitizen"]  = inputs["SeniorCitizen"]
    row["tenure"]         = inputs["tenure"]
    row["MonthlyCharges"] = inputs["MonthlyCharges"]
    row["TotalCharges"]   = inputs["TotalCharges"]
    for col, val in inputs.items():
        if col in CAT_OPTS:
            key = f"{col}_{val}"
            if key in row:
                row[key] = 1
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔮 AI-Powered · Logistic Regression</div>
  <h1 class="hero-title">ChurnGuard</h1>
  <p class="hero-sub">Customer retention intelligence — predict churn risk before it's too late</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Input form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                color:#e8e8f4;margin-bottom:1.5rem;letter-spacing:.02em;">
        ⚙ Customer Profile
    </div>
    """, unsafe_allow_html=True)

    # ── Demographics ──
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">👤 Demographics</div>', unsafe_allow_html=True)
    gender    = st.selectbox("Gender", CAT_OPTS["gender"])
    senior    = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    partner   = st.selectbox("Has Partner", CAT_OPTS["Partner"])
    dependents= st.selectbox("Has Dependents", CAT_OPTS["Dependents"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Account info ──
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📋 Account Info</div>', unsafe_allow_html=True)
    tenure    = st.slider("Tenure (months)", 0, 72, 12)
    contract  = st.selectbox("Contract Type", CAT_OPTS["Contract"])
    paperless = st.selectbox("Paperless Billing", CAT_OPTS["PaperlessBilling"])
    payment   = st.selectbox("Payment Method", CAT_OPTS["PaymentMethod"])
    monthly   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total_c   = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure) or 0.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Phone services ──
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📞 Phone Services</div>', unsafe_allow_html=True)
    phone     = st.selectbox("Phone Service", CAT_OPTS["PhoneService"])
    multiline = st.selectbox("Multiple Lines", CAT_OPTS["MultipleLines"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Internet services ──
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">🌐 Internet Services</div>', unsafe_allow_html=True)
    internet  = st.selectbox("Internet Service", CAT_OPTS["InternetService"])
    online_sec= st.selectbox("Online Security", CAT_OPTS["OnlineSecurity"])
    online_bk = st.selectbox("Online Backup", CAT_OPTS["OnlineBackup"])
    device_pr = st.selectbox("Device Protection", CAT_OPTS["DeviceProtection"])
    tech_sup  = st.selectbox("Tech Support", CAT_OPTS["TechSupport"])
    streaming_tv  = st.selectbox("Streaming TV", CAT_OPTS["StreamingTV"])
    streaming_mv  = st.selectbox("Streaming Movies", CAT_OPTS["StreamingMovies"])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔮  Predict Churn Risk")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
churn_rate   = (df["Churn"] == "Yes").mean() * 100
avg_tenure   = df["tenure"].mean()
avg_monthly  = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card accent">
      <div class="metric-label">Total Customers</div>
      <div class="metric-value accent">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card danger">
      <div class="metric-label">Overall Churn Rate</div>
      <div class="metric-value danger">{churn_rate:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card safe">
      <div class="metric-label">Avg Tenure</div>
      <div class="metric-value safe">{avg_tenure:.0f} mo</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card accent">
      <div class="metric-label">Avg Monthly Charges</div>
      <div class="metric-value accent">${avg_monthly:.0f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — Prediction + Charts
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.05, 1], gap="large")

# ── RIGHT: Analytics charts ────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-title"><span></span>Dataset Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  Churn by Contract  ", "  Monthly Charges  ", "  Tenure Distribution  "])

    PLOT_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#6b6b88", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="#222230", linecolor="#222230"),
        yaxis=dict(gridcolor="#222230", linecolor="#222230"),
        height=280,
    )

    with tab1:
        ct = df.groupby(["Contract","Churn"]).size().unstack(fill_value=0).reset_index()
        fig = go.Figure()
        fig.add_bar(x=ct["Contract"], y=ct.get("Yes", pd.Series([0]*len(ct))),
                    name="Churned", marker_color="#ff4560", marker_line_width=0)
        fig.add_bar(x=ct["Contract"], y=ct.get("No", pd.Series([0]*len(ct))),
                    name="Retained", marker_color="#7c5cfc", marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, barmode="group",
                          legend=dict(bgcolor="rgba(0,0,0,0)", x=.7, y=1))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        for label, color in [("Yes","#ff4560"),("No","#7c5cfc")]:
            sub = df[df["Churn"]==label]["MonthlyCharges"]
            fig2.add_trace(go.Histogram(x=sub, name=label, marker_color=color,
                                        opacity=.75, nbinsx=25))
        fig2.update_layout(**PLOT_LAYOUT, barmode="overlay",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        for label, color in [("Yes","#ff4560"),("No","#7c5cfc")]:
            sub = df[df["Churn"]==label]["tenure"]
            fig3.add_trace(go.Histogram(x=sub, name=label, marker_color=color,
                                        opacity=.75, nbinsx=25))
        fig3.update_layout(**PLOT_LAYOUT, barmode="overlay",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, use_container_width=True)


# ── LEFT: Prediction result ────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-title"><span></span>Churn Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        inputs = dict(
            gender=gender, SeniorCitizen=senior, Partner=partner,
            Dependents=dependents, tenure=tenure, PhoneService=phone,
            MultipleLines=multiline, InternetService=internet,
            OnlineSecurity=online_sec, OnlineBackup=online_bk,
            DeviceProtection=device_pr, TechSupport=tech_sup,
            StreamingTV=streaming_tv, StreamingMovies=streaming_mv,
            Contract=contract, PaperlessBilling=paperless,
            PaymentMethod=payment, MonthlyCharges=monthly,
            TotalCharges=total_c,
        )
        X   = build_input(inputs)
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        churn = bool(pred == 1)

        # ── Gauge ──
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "family": "Syne", "color": "#ff4560" if churn else "#00e5a0"}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#333", "tickfont": {"color":"#6b6b88","size":10}},
                "bar":  {"color": "#ff4560" if churn else "#00e5a0", "thickness": .25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,40],   "color": "rgba(0,229,160,.08)"},
                    {"range": [40,70],  "color": "rgba(255,193,7,.08)"},
                    {"range": [70,100], "color": "rgba(255,69,96,.08)"},
                ],
                "threshold": {"line": {"color":"#fff","width":2}, "thickness":.8, "value": prob*100},
            },
            domain={"x":[0,1],"y":[0,1]},
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#e8e8f4"),
            height=240, margin=dict(l=20,r=20,t=20,b=0),
        )
        st.plotly_chart(gauge, use_container_width=True)

        # ── Result banner ──
        if churn:
            st.markdown(f"""
            <div class="result-banner churn">
              <div class="result-icon">⚠️</div>
              <div>
                <div class="result-heading churn">High Churn Risk</div>
                <div class="result-sub">This customer has a <strong style="color:#ff4560">{prob*100:.1f}%</strong> probability of churning.
                Immediate retention action is recommended.</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner safe">
              <div class="result-icon">✅</div>
              <div>
                <div class="result-heading safe">Low Churn Risk</div>
                <div class="result-sub">This customer has only a <strong style="color:#00e5a0">{prob*100:.1f}%</strong> probability of churning.
                They appear to be satisfied and loyal.</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Risk breakdown ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>Risk Factor Breakdown</div>', unsafe_allow_html=True)

        risk_factors = {
            "Month-to-month contract": 1 if contract == "Month-to-month" else 0,
            "Short tenure (< 12 mo)":  1 if tenure < 12 else 0,
            "High monthly charges":    1 if monthly > 70 else 0,
            "No online security":      1 if online_sec == "No" else 0,
            "Electronic check payment":1 if payment == "Electronic check" else 0,
            "Fiber optic internet":    1 if internet == "Fiber optic" else 0,
        }
        for factor, val in risk_factors.items():
            cols = st.columns([3, 1])
            cols[0].markdown(f"<span style='font-size:.85rem;color:{'#ff4560' if val else '#6b6b88'}'>"
                             f"{'🔴' if val else '🟢'} {factor}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span style='font-size:.85rem;color:{'#ff4560' if val else '#00e5a0'};font-weight:600'>"
                             f"{'Risk' if val else 'OK'}</span>", unsafe_allow_html=True)

    else:
        # placeholder state
        st.markdown("""
        <div style="background:var(--card);border:1px dashed var(--border);border-radius:16px;
                    padding:3rem 2rem;text-align:center;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#e8e8f4;margin-bottom:.5rem">
            Ready to Predict
          </div>
          <div style="color:var(--muted);font-size:.85rem;max-width:260px;margin:0 auto">
            Configure the customer profile in the sidebar and click <strong style="color:#a78bfa">Predict Churn Risk</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show model info
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card accent">
              <div class="metric-label">Model Type</div>
              <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#a78bfa;margin-top:.3rem">
                Logistic Regression
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card accent">
              <div class="metric-label">Input Features</div>
              <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#a78bfa;margin-top:.3rem">
                {len(FEATURE_COLS)} encoded
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("📊  Explore Dataset", expanded=False):
    col_a, col_b = st.columns([3,1])
    with col_b:
        churn_filter = st.selectbox("Filter by Churn", ["All","Yes","No"], key="df_filter")
    dff = df if churn_filter == "All" else df[df["Churn"] == churn_filter]
    st.dataframe(
        dff.head(50),
        use_container_width=True,
        height=300,
        hide_index=True,
    )
    st.caption(f"Showing {min(50, len(dff))} of {len(dff):,} rows  ·  {df.shape[1]} columns")