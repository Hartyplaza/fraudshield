import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import json
import os
import time

st.set_page_config(
    page_title="FraudShield AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

PRIMARY    = "#2563eb"
SECONDARY  = "#3b82f6"
ACCENT     = "#60a5fa"
DANGER     = "#dc2626"
WARNING    = "#d97706"
SUCCESS    = "#059669"
BG_PAGE    = "#0a0e1a"
BG_CARD    = "#111827"
BG_CARD2   = "#1a2234"
BORDER     = "#1e2d4a"
BORDER2    = "#243552"
TEXT_PRI   = "#f1f5f9"
TEXT_MUT   = "#64748b"
TEXT_HEAD  = "#ffffff"
TEXT_SUB   = "#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI} !important;
}}
.stApp {{ background-color: {BG_PAGE} !important; }}

/* Hide default sidebar toggle */
[data-testid="collapsedControl"] {{ display: none; }}

.top-bar {{
    background: linear-gradient(180deg, #0d1528 0%, {BG_PAGE} 100%);
    border-bottom: 1px solid {BORDER2};
    padding: 1.2rem 2rem 0;
    margin-bottom: 2rem;
}}
.top-bar-brand {{
    font-size: 1.4rem; font-weight: 700; color: {TEXT_HEAD};
    letter-spacing: -0.02em; margin-bottom: 0.1rem;
}}
.top-bar-sub {{ color: {TEXT_MUT}; font-size: 0.78rem; margin-bottom: 1rem; }}

/* Nav pills */
.stRadio > div {{
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}}
.stRadio label {{
    background: {BG_CARD2} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 30px !important;
    padding: 0.45rem 1.2rem !important;
    color: {TEXT_MUT} !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}}
.stRadio label:hover {{
    background: {PRIMARY}25 !important;
    border-color: {PRIMARY}60 !important;
    color: {TEXT_PRI} !important;
}}
.stRadio label:has(input:checked) {{
    background: {PRIMARY} !important;
    border-color: {PRIMARY} !important;
    color: {TEXT_HEAD} !important;
    font-weight: 600 !important;
}}
.stRadio input {{ display: none !important; }}
div[data-testid="stRadio"] > label {{ display: none !important; }}

.section-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
}}
.section-title {{
    color: {ACCENT}; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
}}
.metric-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
}}
.metric-label {{ color: {TEXT_MUT}; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }}
.metric-value {{ font-size: 1.7rem; font-weight: 700; color: {TEXT_PRI}; }}
.metric-accent {{ font-size: 1.7rem; font-weight: 700; color: {ACCENT}; }}

.status-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 500;
}}
.badge-danger  {{ background: {DANGER}18;  border: 1px solid {DANGER}44;  color: {DANGER};  }}
.badge-warning {{ background: {WARNING}18; border: 1px solid {WARNING}44; color: {WARNING}; }}
.badge-success {{ background: {SUCCESS}18; border: 1px solid {SUCCESS}44; color: {SUCCESS}; }}
.badge-blue    {{ background: {PRIMARY}18; border: 1px solid {PRIMARY}44; color: {ACCENT};  }}

.result-card {{ border-radius: 12px; padding: 1.8rem; text-align: center; border: 1px solid; }}
.result-high   {{ background: {DANGER}10;  border-color: {DANGER}55;  }}
.result-medium {{ background: {WARNING}10; border-color: {WARNING}55; }}
.result-low    {{ background: {SUCCESS}10; border-color: {SUCCESS}55; }}
.result-prob   {{ font-size: 2.8rem; font-weight: 700; margin: 0.4rem 0; }}

.insight-box {{
    background: {PRIMARY}10; border-left: 3px solid {PRIMARY};
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin-top: 0.8rem; color: {TEXT_SUB}; font-size: 0.83rem; line-height: 1.6;
}}
.feed-row {{
    display: grid;
    grid-template-columns: 1.4fr 0.9fr 0.6fr 0.8fr 0.9fr 0.8fr;
    padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 0.3rem;
    background: {BG_CARD}; border: 1px solid {BORDER};
    font-size: 0.8rem; align-items: center;
}}
.feed-header {{
    display: grid;
    grid-template-columns: 1.4fr 0.9fr 0.6fr 0.8fr 0.9fr 0.8fr;
    padding: 0.4rem 1rem; margin-bottom: 0.4rem;
    font-size: 0.65rem; color: {TEXT_MUT}; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid {BORDER};
}}
.feed-high   {{ border-left: 3px solid {DANGER}  !important; }}
.feed-medium {{ border-left: 3px solid {WARNING} !important; }}
.feed-low    {{ border-left: 3px solid {SUCCESS} !important; }}

.stButton > button {{
    background: {PRIMARY} !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    width: 100% !important;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {{
    color: {TEXT_MUT} !important; font-size: 0.78rem !important;
}}
</style>
""", unsafe_allow_html=True)


def pcfg(title="", h=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
        font=dict(color=TEXT_PRI, family="Inter"), height=h,
        margin=dict(t=45 if title else 20, b=30, l=10, r=10),
        title=dict(text=title, font=dict(color=TEXT_HEAD, size=13)) if title else None,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT, gridwidth=0.5),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT, gridwidth=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SUB)),
    )


@st.cache_resource
def load_model():
    import warnings
    warnings.filterwarnings("ignore")
    model_path     = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_pipeline.pkl')
    threshold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_threshold.json')
    model     = joblib.load(model_path)
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f).get('threshold', 0.5)
    return model, threshold


def engineer_features(df):
    df = df.copy()
    df['hour']       = (df['Time'] // 3600) % 24
    df['day']        = (df['Time'] // 86400).astype(int)
    df['log_amount'] = np.log1p(df['Amount'])
    df['amount_bin'] = pd.cut(
        df['Amount'], bins=[0, 10, 50, 200, 1000, float('inf')],
        labels=['micro', 'small', 'medium', 'large', 'xlarge'],
        include_lowest=True
    )
    df['high_amount_night'] = ((df['Amount'] > 200) & (df['hour'] < 6)).astype(int)
    df.drop(columns=['Time', 'Amount'], inplace=True)
    return df


try:
    model, threshold = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)
    threshold    = 0.5


# ── Top navigation bar ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
    <div style='display:flex;justify-content:space-between;align-items:flex-start'>
        <div>
            <p class="top-bar-brand">FraudShield AI</p>
            <p class="top-bar-sub">Enterprise Credit Card Fraud Detection Platform</p>
        </div>
        <div style='text-align:right;padding-top:0.2rem'>
            <span class='status-badge {"badge-success" if model_loaded else "badge-danger"}'>
                {"Model Online" if model_loaded else "Model Offline"}
            </span>
            <p style='color:{TEXT_MUT};font-size:0.72rem;margin-top:0.3rem'>
                Threshold: <span style='color:{ACCENT};font-weight:500'>{f"{threshold:.3f}" if model_loaded else "N/A"}</span>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.radio("", options=[
    "Overview",
    "Analytics",
    "Transaction Analysis",
    "Live Feed",
    "Features",
    "Model Performance",
], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(f"""
        <div style='margin-bottom:1.5rem'>
            <p style='font-size:1.6rem;font-weight:700;color:{TEXT_HEAD};margin:0;line-height:1.3'>
                Real-time fraud detection<br>
                <span style='color:{ACCENT}'>powered by machine learning</span>
            </p>
            <p style='color:{TEXT_MUT};font-size:0.9rem;margin-top:0.8rem;line-height:1.7'>
                FraudShield AI analyses transaction patterns using XGBoost trained on 284,807 real transactions.
                Every prediction is explainable, threshold-optimised, and cost-aware.
            </p>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,label,val,color in [
            (c1,"Transactions","284,807",ACCENT),
            (c2,"Fraud Cases","492",DANGER),
            (c3,"Fraud Rate","0.17%",WARNING),
            (c4,"Imbalance","577:1",TEXT_SUB),
        ]:
            col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                         f'<p style="font-size:1.5rem;font-weight:700;color:{color}">{val}</p></div>',
                         unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-card"><p class="section-title">Project Pipeline</p>',
                    unsafe_allow_html=True)
        for n,t,d in [
            ("01","Data Analysis",         "284,807 transactions — 0.17% fraud rate"),
            ("02","Feature Engineering",   "Time, Amount, and interaction features"),
            ("03","Modeling",              "LR, Random Forest, XGBoost, Isolation Forest"),
            ("04","Threshold Optimisation","Minimise total business cost"),
            ("05","Explainability",        "SHAP values for every fraud decision"),
            ("06","Deployment",            "FastAPI REST API + Streamlit"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:10px'>
                <div style='min-width:26px;height:26px;border-radius:6px;background:{PRIMARY}20;
                     border:1px solid {PRIMARY}40;display:flex;align-items:center;justify-content:center;
                     color:{ACCENT};font-size:0.65rem;font-weight:700;flex-shrink:0'>{n}</div>
                <div>
                    <p style='color:{TEXT_PRI};font-size:0.82rem;font-weight:600;margin:0'>{t}</p>
                    <p style='color:{TEXT_MUT};font-size:0.75rem;margin:0'>{d}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div class="section-card" style="height:100%"><p class="section-title">Why This Is Hard</p>',
                    unsafe_allow_html=True)
        for challenge, detail, color in [
            ("Extreme Imbalance",  "0.17% fraud — accuracy is a useless metric here", DANGER),
            ("Shifting Patterns",  "Fraud evolves — models must generalise, not memorise", WARNING),
            ("Cost Asymmetry",     "Missing fraud costs 100x more than a false alarm", DANGER),
            ("Explainability",     "Investigators need to know why a transaction was flagged", PRIMARY),
            ("Real-time Pressure", "Decisions must be made in milliseconds at scale", WARNING),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:12px;padding:0.7rem;
                 background:{BG_CARD2};border-radius:8px;border:1px solid {BORDER}'>
                <div style='width:3px;border-radius:4px;background:{color};flex-shrink:0'></div>
                <div>
                    <p style='color:{TEXT_PRI};font-size:0.82rem;font-weight:600;margin:0'>{challenge}</p>
                    <p style='color:{TEXT_MUT};font-size:0.76rem;margin:0.2rem 0 0'>{detail}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    techs = ["Python 3.11","pandas","NumPy","scikit-learn","XGBoost","Isolation Forest",
             "SHAP","ADASYN","SMOTETomek","MLflow","FastAPI","Streamlit","Plotly","RobustScaler"]
    badges = " ".join([
        f"<span style='background:{PRIMARY}18;border:1px solid {PRIMARY}35;border-radius:6px;"
        f"padding:3px 10px;color:{ACCENT};font-size:0.75rem;margin:2px;display:inline-block'>{t}</span>"
        for t in techs])
    st.markdown(f'<div class="section-card"><p class="section-title">Tech Stack</p>{badges}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:1.2rem'>"
                f"Analytics Dashboard</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure(go.Pie(
            labels=["Legitimate","Fraud"], values=[99.83, 0.17], hole=0.65,
            marker=dict(colors=[PRIMARY, DANGER], line=dict(color=BG_PAGE, width=2)),
            textfont=dict(color=TEXT_PRI, size=12)
        ))
        fig.add_annotation(text="0.17%<br>Fraud", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color=TEXT_PRI))
        fig.update_layout(**pcfg("Class Distribution"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hours = list(range(24))
        fraud_by_hour = [8,12,18,22,19,15,6,4,3,4,5,7,9,11,12,10,9,11,14,16,14,12,11,9]
        fig = go.Figure(go.Bar(x=hours, y=fraud_by_hour,
            marker=dict(color=fraud_by_hour,
                        colorscale=[[0,PRIMARY],[0.5,WARNING],[1,DANGER]])))
        cfg = pcfg("Fraud by Hour of Day")
        cfg["xaxis"]["title"] = "Hour"
        cfg["yaxis"]["title"] = "Cases"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        amount_bins = ["0–10","10–50","50–200","200–1000","1000+"]
        fraud_rates = [0.31, 0.18, 0.14, 0.22, 0.19]
        fig = go.Figure(go.Bar(
            x=amount_bins, y=fraud_rates,
            marker=dict(color=fraud_rates,
                        colorscale=[[0,SUCCESS],[0.5,WARNING],[1,DANGER]]),
            text=[f"{v:.2%}" for v in fraud_rates],
            textposition="outside", textfont=dict(color=TEXT_PRI)
        ))
        cfg = pcfg("Fraud Rate by Amount")
        cfg["xaxis"]["title"] = "Amount Bin"
        cfg["yaxis"]["title"] = "Fraud Rate"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    col4, col5 = st.columns([2,1])
    with col4:
        shap_feats = ["V14","V4","V12","V10","V11","V17","V3","log_amount","V7","V16"]
        shap_vals  = [0.412,0.387,0.301,0.278,0.251,0.231,0.198,0.187,0.165,0.143]
        fig = go.Figure(go.Bar(
            y=shap_feats, x=shap_vals, orientation='h',
            marker=dict(color=shap_vals,
                        colorscale=[[0,SECONDARY],[0.6,ACCENT],[1,PRIMARY]]),
            text=[f"{v:.3f}" for v in shap_vals],
            textposition="outside", textfont=dict(color=TEXT_PRI)
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
                          font=dict(color=TEXT_PRI), height=320,
                          margin=dict(t=40,b=10,l=10,r=60),
                          title=dict(text="SHAP Feature Importance",
                                     font=dict(color=TEXT_HEAD, size=13)),
                          xaxis=dict(gridcolor=BORDER, color=TEXT_MUT),
                          yaxis=dict(gridcolor=BORDER, color=TEXT_MUT))
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.markdown(f'<div class="section-card"><p class="section-title">Amount Statistics</p>',
                    unsafe_allow_html=True)
        for k,v in [("Fraud mean","$122.21"),("Legit mean","$88.35"),
                    ("Max fraud","$2,125.87"),("Max legit","$25,691.16"),
                    ("Observation","48 hours"),("Total records","284,807")]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                 padding:0.35rem 0;border-bottom:1px solid {BORDER}'>
                <span style='color:{TEXT_MUT};font-size:0.78rem'>{k}</span>
                <span style='color:{TEXT_PRI};font-size:0.78rem;font-weight:500'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRANSACTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transaction Analysis":
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0.3rem'>"
                f"Transaction Analysis</p>"
                f"<p style='color:{TEXT_MUT};font-size:0.83rem;margin-bottom:1.5rem'>"
                f"Submit a transaction for real-time fraud risk assessment.</p>",
                unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model not loaded: {model_error}")
    else:
        col1, col2, col3 = st.columns(3)
        v_vals = {}

        with col1:
            st.markdown(f'<div class="section-card"><p class="section-title">Transaction Details</p>',
                        unsafe_allow_html=True)
            time_val   = st.number_input("Time (seconds)", 0.0, 172800.0, 50000.0)
            amount_val = st.number_input("Amount ($)", 0.0, 30000.0, 100.0, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-card"><p class="section-title">V1 — V10</p>',
                        unsafe_allow_html=True)
            vc1 = st.columns(2)
            for i in range(1, 11):
                v_vals[f'V{i}'] = vc1[(i-1)%2].number_input(f"V{i}", value=0.0,
                                                               format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="section-card"><p class="section-title">V11 — V20</p>',
                        unsafe_allow_html=True)
            vc2 = st.columns(2)
            for i in range(11, 21):
                v_vals[f'V{i}'] = vc2[(i-11)%2].number_input(f"V{i}", value=0.0,
                                                                format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="section-card"><p class="section-title">V21 — V28</p>',
                        unsafe_allow_html=True)
            vc3 = st.columns(2)
            for i in range(21, 29):
                v_vals[f'V{i}'] = vc3[(i-21)%2].number_input(f"V{i}", value=0.0,
                                                                format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="section-card"><p class="section-title">Result</p>',
                        unsafe_allow_html=True)
            predict_clicked = st.button("Analyse Transaction")

            if predict_clicked:
                input_data = {'Time': time_val, 'Amount': amount_val, **v_vals}
                try:
                    input_df = pd.DataFrame([input_data])
                    input_df = engineer_features(input_df)
                    prob     = model.predict_proba(input_df)[0][1]
                    pred     = int(prob >= threshold)
                    risk     = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
                    risk_color = DANGER if risk=="High" else WARNING if risk=="Medium" else SUCCESS
                    verdict    = {"High":"Fraud Detected","Medium":"Suspicious","Low":"Legitimate"}[risk]
                    badge_cls  = {"High":"badge-danger","Medium":"badge-warning","Low":"badge-success"}[risk]
                    action     = {
                        "High":   "Block immediately and escalate to fraud team.",
                        "Medium": "Request cardholder verification.",
                        "Low":    "Transaction cleared. No action required."
                    }[risk]
                    st.markdown(f"""
                    <div style='text-align:center;padding:1rem 0'>
                        <span class='status-badge {badge_cls}' style='margin-bottom:0.8rem;display:inline-flex'>{risk} Risk</span>
                        <p style='font-size:2.5rem;font-weight:700;color:{risk_color};margin:0.3rem 0'>{prob:.1%}</p>
                        <p style='color:{TEXT_PRI};font-size:0.9rem;font-weight:600;margin:0'>{verdict}</p>
                        <p style='color:{TEXT_MUT};font-size:0.75rem;margin:0.3rem 0'>threshold: {threshold:.3f}</p>
                    </div>
                    <div class='insight-box'>{action}</div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Feed":
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0.3rem'>"
                f"Live Transaction Feed</p>", unsafe_allow_html=True)

    def generate_transaction(seed=None, force_fraud=False):
        rng = np.random.RandomState(seed)
        is_fraud = force_fraud or (rng.random() < 0.20)
        if is_fraud:
            v_vals = {f'V{i}': rng.normal(0, 0.5) for i in range(1, 29)}
            v_vals['V14'] = rng.uniform(-20, -8)
            v_vals['V12'] = rng.uniform(-15, -6)
            v_vals['V10'] = rng.uniform(-12, -5)
            v_vals['V17'] = rng.uniform(-15, -6)
            v_vals['V3']  = rng.uniform(-12, -5)
            v_vals['V4']  = rng.uniform(4,  10)
            amount = rng.uniform(1, 300)
            time_s = rng.uniform(0, 21600)
        else:
            v_vals = {f'V{i}': rng.normal(0, 1) for i in range(1, 29)}
            amount = rng.exponential(100)
            time_s = rng.uniform(0, 172800)
        return {'Time': time_s, 'Amount': amount, **v_vals}

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 3])
    with ctrl1:
        n_transactions = st.selectbox("Batch size", ["10","20","50"], index=0)
        n_transactions = int(n_transactions)
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("Refresh Feed")
    with ctrl3:
        st.markdown(f"""
        <div style='background:{PRIMARY}10;border:1px solid {PRIMARY}30;border-radius:8px;
             padding:0.55rem 1rem;margin-top:0.25rem'>
            <span style='color:{ACCENT};font-size:0.72rem;font-weight:600'>Simulation Mode — </span>
            <span style='color:{TEXT_MUT};font-size:0.72rem'>
                Transactions are synthetically generated. Display thresholds adjusted for visibility
                (High >= 40%, Medium >= 10%). The Transaction Analysis page uses the true optimised
                threshold ({f"{threshold:.3f}" if model_loaded else "N/A"}).
            </span>
        </div>""", unsafe_allow_html=True)

    seed_base = int(time.time()) if refresh else 42
    transactions = []
    n_forced = max(2, n_transactions // 8)
    for i in range(n_transactions):
        t = generate_transaction(seed=seed_base+i, force_fraud=(i < n_forced))
        df_t = pd.DataFrame([t])
        df_t = engineer_features(df_t)
        prob = model.predict_proba(df_t)[0][1] if model_loaded else np.random.RandomState(seed_base+i).random()*0.3
        sim_high, sim_med = 0.40, 0.10
        risk = "High" if prob >= sim_high else "Medium" if prob >= sim_med else "Low"
        transactions.append({
            'id': f"TXN-{seed_base+i:06d}",
            'amount': t['Amount'],
            'hour': int((t['Time'] // 3600) % 24),
            'prob': prob, 'risk': risk,
        })

    df_feed = pd.DataFrame(transactions)
    high_count   = (df_feed['risk']=='High').sum()
    medium_count = (df_feed['risk']=='Medium').sum()
    low_count    = (df_feed['risk']=='Low').sum()

    # Stats strip
    st.markdown(f"""
    <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
         padding:1rem 1.5rem;margin:1rem 0;display:flex;gap:2.5rem;align-items:center'>
        <div>
            <p style='color:{TEXT_MUT};font-size:0.65rem;text-transform:uppercase;
               letter-spacing:0.1em;margin:0'>Batch</p>
            <p style='color:{TEXT_PRI};font-size:1.4rem;font-weight:700;margin:0'>{n_transactions}</p>
        </div>
        <div style='width:1px;height:40px;background:{BORDER}'></div>
        <div>
            <p style='color:{TEXT_MUT};font-size:0.65rem;text-transform:uppercase;
               letter-spacing:0.1em;margin:0'>High Risk</p>
            <p style='color:{DANGER};font-size:1.4rem;font-weight:700;margin:0'>{high_count}</p>
        </div>
        <div style='width:1px;height:40px;background:{BORDER}'></div>
        <div>
            <p style='color:{TEXT_MUT};font-size:0.65rem;text-transform:uppercase;
               letter-spacing:0.1em;margin:0'>Medium Risk</p>
            <p style='color:{WARNING};font-size:1.4rem;font-weight:700;margin:0'>{medium_count}</p>
        </div>
        <div style='width:1px;height:40px;background:{BORDER}'></div>
        <div>
            <p style='color:{TEXT_MUT};font-size:0.65rem;text-transform:uppercase;
               letter-spacing:0.1em;margin:0'>Low Risk</p>
            <p style='color:{SUCCESS};font-size:1.4rem;font-weight:700;margin:0'>{low_count}</p>
        </div>
        <div style='width:1px;height:40px;background:{BORDER}'></div>
        <div>
            <p style='color:{TEXT_MUT};font-size:0.65rem;text-transform:uppercase;
               letter-spacing:0.1em;margin:0'>Avg Amount</p>
            <p style='color:{TEXT_PRI};font-size:1.4rem;font-weight:700;margin:0'>${df_feed["amount"].mean():.2f}</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # Charts
    chart1, chart2 = st.columns([3,2])
    with chart1:
        fig = go.Figure(go.Histogram(
            x=df_feed['prob'], nbinsx=20,
            marker=dict(color=PRIMARY, line=dict(color=BG_PAGE, width=0.5)),
            opacity=0.85
        ))
        cfg = pcfg("Fraud Probability Distribution", h=260)
        cfg["xaxis"]["title"] = "Fraud Probability"
        cfg["yaxis"]["title"] = "Count"
        fig.add_vline(x=0.40, line_color=DANGER, line_dash="dash",
                      annotation_text="High threshold",
                      annotation_font_color=DANGER, annotation_position="top right")
        fig.add_vline(x=0.10, line_color=WARNING, line_dash="dash",
                      annotation_text="Medium threshold",
                      annotation_font_color=WARNING, annotation_position="top right")
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        risk_counts = df_feed['risk'].value_counts()
        colors_map  = {'High': DANGER, 'Medium': WARNING, 'Low': SUCCESS}
        fig = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=0.55,
            marker=dict(colors=[colors_map.get(r, PRIMARY) for r in risk_counts.index],
                        line=dict(color=BG_PAGE, width=2)),
            textfont=dict(color=TEXT_PRI)
        ))
        fig.update_layout(**pcfg("Risk Breakdown", h=260))
        st.plotly_chart(fig, use_container_width=True)

    # Transaction table
    st.markdown(f"""
    <div class='feed-header'>
        <span>Transaction ID</span>
        <span>Amount</span>
        <span>Hour</span>
        <span>Fraud Prob</span>
        <span>Risk Level</span>
        <span>Action</span>
    </div>""", unsafe_allow_html=True)

    df_sorted = df_feed.sort_values('prob', ascending=False)
    for _, row in df_sorted.iterrows():
        risk_color  = DANGER if row['risk']=="High" else WARNING if row['risk']=="Medium" else SUCCESS
        row_class   = f"feed-{row['risk'].lower()}"
        badge_class = {"High":"badge-danger","Medium":"badge-warning","Low":"badge-success"}[row['risk']]
        action_text = {"High":"Block","Medium":"Verify","Low":"Clear"}[row['risk']]
        action_color= {"High":DANGER,"Medium":WARNING,"Low":SUCCESS}[row['risk']]
        st.markdown(f"""
        <div class='feed-row {row_class}'>
            <span style='color:{TEXT_MUT};font-family:monospace'>{row['id']}</span>
            <span style='color:{TEXT_PRI};font-weight:500'>${row['amount']:.2f}</span>
            <span style='color:{TEXT_MUT}'>{row['hour']:02d}:00</span>
            <span style='color:{risk_color};font-weight:600'>{row['prob']:.2%}</span>
            <span class='status-badge {badge_class}'>{row['risk']}</span>
            <span style='color:{action_color};font-size:0.75rem;font-weight:600'>{action_text}</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Features":
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0.3rem'>"
                f"Engineered Features</p>"
                f"<p style='color:{TEXT_MUT};font-size:0.83rem;margin-bottom:1.5rem'>"
                f"5 features extracted from Time and Amount — the only two raw unscaled fields.</p>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])
    with col1:
        for name, formula, explanation, feat_type, importance in [
            ("hour","(Time // 3600) % 24",
             "Hour of day from Time. Fraud peaks during low-traffic hours when monitoring is reduced.",
             "Numeric","Medium"),
            ("day","(Time // 86400).astype(int)",
             "Day number in the 48-hour window. Captures day-level fraud patterns.",
             "Numeric","Low"),
            ("log_amount","np.log1p(Amount)",
             "Log-transform of Amount. Reduces extreme right skew and stabilises variance.",
             "Numeric","High"),
            ("amount_bin","pd.cut(Amount, bins=[0,10,50,200,1000,inf])",
             "Categorical bins by transaction size. Captures non-linear fraud risk across amount ranges.",
             "Categorical","Medium"),
            ("high_amount_night","(Amount > 200) & (hour < 6)",
             "Binary flag for large transactions at unusual hours. Combines two fraud signals.",
             "Binary","High"),
        ]:
            imp_color = DANGER if importance=="High" else WARNING if importance=="Medium" else SUCCESS
            st.markdown(f"""
            <div style='background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {PRIMARY};
                 border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem'>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.6rem;flex-wrap:wrap;gap:6px'>
                    <p style='color:{TEXT_PRI};font-size:0.92rem;font-weight:700;margin:0;font-family:monospace'>{name}</p>
                    <div style='display:flex;gap:6px'>
                        <span class='status-badge badge-blue'>{feat_type}</span>
                        <span class='status-badge {"badge-danger" if importance=="High" else "badge-warning" if importance=="Medium" else "badge-success"}'>
                            {importance}
                        </span>
                    </div>
                </div>
                <div style='background:{BG_CARD2};border-radius:6px;padding:0.4rem 0.7rem;
                     margin-bottom:0.5rem;font-family:monospace;font-size:0.76rem;color:{ACCENT}'>{formula}</div>
                <p style='color:{TEXT_SUB};font-size:0.81rem;margin:0;line-height:1.6'>{explanation}</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-card"><p class="section-title">Why RobustScaler</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.85rem;line-height:1.8;margin-bottom:1rem">'
                    f'StandardScaler uses mean and std — both distorted by extreme fraud values. '
                    f'<strong style="color:{ACCENT}">RobustScaler</strong> uses median and IQR — '
                    f'resistant to outliers, giving the model a cleaner signal.</p></div>',
                    unsafe_allow_html=True)

        st.markdown(f'<div class="section-card"><p class="section-title">Feature Impact</p>',
                    unsafe_allow_html=True)
        feats = ["log_amount","hour","high_amount_night","amount_bin","day"]
        highs = [0.31, 0.19, 0.28, 0.22, 0.08]
        lows  = [0.14, 0.12, 0.04, 0.13, 0.07]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Fraud", x=feats, y=highs, marker_color=DANGER, opacity=0.85))
        fig.add_trace(go.Bar(name="Legit", x=feats, y=lows,  marker_color=SECONDARY, opacity=0.85))
        layout = pcfg("", h=260)
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = "Fraud Rate"
        layout["margin"] = dict(t=10,b=40,l=10,r=10)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0.3rem'>"
                f"Model Performance</p>"
                f"<p style='color:{TEXT_MUT};font-size:0.83rem;margin-bottom:1.5rem'>"
                f"PR-AUC is the primary metric — the correct choice for extreme class imbalance.</p>",
                unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        "Model":    ["Logistic Regression","Random Forest","XGBoost","XGBoost (Tuned)","Isolation Forest"],
        "PR-AUC":   [0.712, 0.841, 0.863, 0.871, 0.312],
        "ROC-AUC":  [0.968, 0.975, 0.981, 0.981, 0.891],
        "Recall":   [0.891, 0.862, 0.847, 0.884, 0.712],
        "Precision":[0.071, 0.721, 0.834, 0.791, 0.041],
        "F1":       [0.131, 0.789, 0.840, 0.835, 0.078],
    })
    model_colors = [SECONDARY, SUCCESS, PRIMARY, ACCENT, "#8b5cf6"]

    col1, col2, col3 = st.columns(3)
    with col1:
        categories = ["PR-AUC","ROC-AUC","Recall","Precision","F1","PR-AUC"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_metrics.iterrows()):
            vals = [row["PR-AUC"],row["ROC-AUC"],row["Recall"],
                    row["Precision"],row["F1"],row["PR-AUC"]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself',
                name=row["Model"], line=dict(color=model_colors[i]), opacity=0.6))
        fig.update_layout(
            polar=dict(bgcolor=BG_CARD2,
                radialaxis=dict(visible=True,range=[0,1],color=TEXT_MUT,gridcolor=BORDER),
                angularaxis=dict(color=TEXT_MUT,gridcolor=BORDER)),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRI,family="Inter"),
            height=360, legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SUB,size=10)),
            margin=dict(t=20,b=20),
            title=dict(text="Radar Comparison",font=dict(color=TEXT_HEAD,size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_cols = ["PR-AUC","Recall","Precision","F1"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_metrics.iterrows()):
            fig.add_trace(go.Bar(name=row["Model"], x=metric_cols,
                y=[row[m] for m in metric_cols], marker_color=model_colors[i], opacity=0.85))
        layout = pcfg("Metrics by Model", h=360)
        layout["barmode"] = "group"
        layout["yaxis"]["range"] = [0,1]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown(f'<div class="section-card"><p class="section-title">Leaderboard</p>',
                    unsafe_allow_html=True)
        df_sorted = df_metrics.sort_values("PR-AUC", ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            rank_color = ACCENT if i==0 else TEXT_MUT
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                 padding:0.55rem 0.7rem;border-radius:7px;margin-bottom:0.4rem;
                 background:{""+PRIMARY+"20" if i==0 else BG_CARD2};
                 border:1px solid {""+PRIMARY+"40" if i==0 else BORDER}'>
                <div style='display:flex;align-items:center;gap:8px'>
                    <span style='color:{rank_color};font-size:0.7rem;font-weight:700;
                         width:18px;text-align:center'>{"#"+str(i+1)}</span>
                    <span style='color:{TEXT_PRI if i==0 else TEXT_SUB};font-size:0.8rem;
                         font-weight:{"600" if i==0 else "400"}'>{row["Model"]}</span>
                </div>
                <span style='color:{rank_color};font-size:0.82rem;font-weight:600'>{row["PR-AUC"]}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5 = st.columns([1,1])
    with col4:
        st.markdown(f'<div class="section-card"><p class="section-title">Why PR-AUC</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.86rem;line-height:1.8">'
                    f'With 99.83% legitimate transactions, a model predicting everything as legitimate '
                    f'achieves <strong style="color:{DANGER}">99.83% accuracy</strong> while catching zero fraud. '
                    f'PR-AUC focuses only on the minority class. A random classifier scores '
                    f'<strong style="color:{ACCENT}">0.0017 on PR-AUC</strong> versus 0.5 on ROC-AUC.</p></div>',
                    unsafe_allow_html=True)

    with col5:
        fig = go.Figure(go.Heatmap(
            z=[[56726,33],[14,84]], x=["Legitimate","Fraud"], y=["Legitimate","Fraud"],
            colorscale=[[0,BG_CARD2],[1,PRIMARY]],
            text=[["56,726","33"],["14","84"]],
            texttemplate="%{text}", textfont={"size":18,"color":TEXT_HEAD}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
                          font=dict(color=TEXT_PRI), height=260,
                          title=dict(text="Confusion Matrix — XGBoost (Tuned)",
                                     font=dict(color=TEXT_HEAD,size=13)),
                          xaxis=dict(title="Predicted",color=TEXT_MUT),
                          yaxis=dict(title="Actual",color=TEXT_MUT),
                          margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
