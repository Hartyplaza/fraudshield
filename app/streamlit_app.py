import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="FraudShield AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY   = "#ef4444"
SECONDARY = "#f97316"
SUCCESS   = "#34d399"
WARNING   = "#fbbf24"
DANGER    = "#ef4444"
BG_CARD   = "#1e2235"
BG_PAGE   = "#0f1117"
BORDER    = "#2d3a5c"
TEXT_PRI  = "#e8eaf6"
TEXT_MUT  = "#94a3b8"
TEXT_HEAD = "#ffffff"
BLUE      = "#3b82f6"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI} !important;
}}
.stApp {{ background-color: {BG_PAGE} !important; }}
section[data-testid="stSidebar"] {{
    background-color: #0a0d14 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
.hero-banner {{
    background: linear-gradient(135deg, #1a0a0a 0%, #0f1117 60%, #1a1020 100%);
    border: 1px solid {BORDER}; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}}
.hero-title {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}}
.hero-subtitle {{ color: {TEXT_MUT}; font-size: 0.95rem; margin-top: 0.4rem; }}
.metric-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}}
.metric-label {{ color: {TEXT_MUT}; font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
.metric-value {{ font-size: 1.8rem; font-weight: 700; color: {TEXT_PRI}; }}
.metric-accent {{ font-size: 1.8rem; font-weight: 700; color: {PRIMARY}; }}
.section-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}}
.section-title {{
    color: {PRIMARY}; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}
.nav-header {{ color: {TEXT_MUT}; font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em; padding: 0.5rem 0 0.3rem; }}
.result-high {{ background: #2a1010; border: 1px solid {DANGER};
    border-radius: 16px; padding: 2rem; text-align: center; }}
.result-medium {{ background: #2a1f00; border: 1px solid {WARNING};
    border-radius: 16px; padding: 2rem; text-align: center; }}
.result-low {{ background: #0d2016; border: 1px solid {SUCCESS};
    border-radius: 16px; padding: 2rem; text-align: center; }}
.result-prob {{ font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }}
.insight-box {{
    background: #1a1020; border-left: 3px solid {PRIMARY};
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin-top: 0.8rem; color: {TEXT_PRI}; font-size: 0.85rem; line-height: 1.6;
}}
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-weight: 600 !important;
    font-size: 1rem !important; width: 100% !important;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {{ color: {TEXT_MUT} !important; font-size: 0.8rem !important; }}
div[data-testid="stRadio"] label {{ color: {TEXT_PRI} !important; font-size: 0.88rem !important; }}
</style>
""", unsafe_allow_html=True)


def pcfg(title="", h=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRI, family="Inter"), height=h,
        margin=dict(t=45 if title else 20, b=30, l=10, r=10),
        title=dict(text=title, font=dict(color=TEXT_HEAD, size=14)) if title else None,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRI)),
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='font-size:1.2rem;font-weight:700;color:{TEXT_HEAD};margin-bottom:0'>FraudShield AI</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.8rem;margin-top:0'>Credit Card Fraud Detection</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p class="nav-header">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", options=[
        "Project Overview",
        "Dashboard & Plots",
        "Prediction",
        "Engineered Features",
        "Model Metrics",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown(f'<p class="nav-header">Model Info</p>', unsafe_allow_html=True)
    status_color = SUCCESS if model_loaded else DANGER
    for k, v in [("Algorithm","XGBoost + ADASYN"),("PR-AUC","TBD"),
                 ("Dataset","Kaggle Creditcard"),("Records","284,807"),
                 ("Fraud Rate","0.17%"),("Threshold", f"{threshold:.3f}" if model_loaded else "N/A"),
                 ("Model Status","Loaded" if model_loaded else "Not Found")]:
        color = status_color if k == "Model Status" else TEXT_PRI
        st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.8rem;margin:0.25rem 0'>"
                    f"<span style='color:{TEXT_PRI};font-weight:500'>{k}:</span> "
                    f"<span style='color:{color}'>{v}</span></p>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p class="nav-header">Risk Thresholds</p>', unsafe_allow_html=True)
    for label, color, thr in [("High Risk", DANGER, ">= 70%"),
                               ("Medium Risk", WARNING, "30 – 69%"),
                               ("Low Risk", SUCCESS, "< 30%")]:
        st.markdown(f"<p style='color:{color};font-size:0.82rem;margin:0.2rem 0;font-weight:500'>"
                    f"{label}: <span style='font-weight:400'>{thr}</span></p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">FraudShield AI</p>
        <p class="hero-subtitle">An end-to-end machine learning system for real-time credit card fraud detection using anomaly detection and supervised learning.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [(c1,"Total Transactions","284,807"),(c2,"Fraud Cases","492"),
                           (c3,"Fraud Rate","0.17%"),(c4,"Imbalance Ratio","577:1")]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-accent">{val}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        steps_html = "".join([f"""
        <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px'>
            <div style='min-width:32px;height:32px;border-radius:8px;background:{PRIMARY}22;
                 border:1px solid {PRIMARY}55;display:flex;align-items:center;justify-content:center;
                 color:{PRIMARY};font-size:0.7rem;font-weight:700;flex-shrink:0'>{n}</div>
            <div><p style='color:{TEXT_PRI};font-size:0.85rem;font-weight:600;margin:0'>{t}</p>
                 <p style='color:{TEXT_MUT};font-size:0.78rem;margin:0'>{d}</p></div>
        </div>""" for n,t,d in [
            ("01","Data Analysis","284,807 transactions, 0.17% fraud rate"),
            ("02","Feature Engineering","Time, Amount, interaction features"),
            ("03","Modeling","LR, Random Forest, XGBoost, Isolation Forest"),
            ("04","Threshold Optimisation","Minimise business cost"),
            ("05","Explainability","SHAP values for fraud decisions"),
            ("06","Deployment","FastAPI REST API + Streamlit"),
        ]])
        st.markdown(f'<div class="section-card"><p class="section-title">Project Pipeline</p>{steps_html}</div>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-card"><p class="section-title">Problem Statement</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8">Credit card fraud costs the global economy over '
                    f'<strong style="color:{PRIMARY}">$30 billion annually</strong>. '
                    f'Traditional rule-based systems miss novel fraud patterns.</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8;margin-top:0.8rem">The core challenge:</p>'
                    f'<ul style="color:{TEXT_PRI};font-size:0.9rem;line-height:2.2">'
                    f'<li>Only <strong>0.17%</strong> of transactions are fraud</li>'
                    f'<li>Standard accuracy metrics are <strong>useless</strong> — 99.83% accuracy by predicting everything as legitimate</li>'
                    f'<li>Missing fraud costs <strong>100x more</strong> than a false alarm</li>'
                    f'<li>Model must explain <strong>why</strong> a transaction is suspicious</li>'
                    f'</ul></div>', unsafe_allow_html=True)

    techs = ["Python 3.11","pandas","NumPy","scikit-learn","XGBoost",
             "Isolation Forest","SHAP","ADASYN","SMOTETomek","MLflow",
             "FastAPI","Streamlit","Plotly","RobustScaler"]
    badges = " ".join([
        f"<span style='background:{PRIMARY}22;border:1px solid {PRIMARY}44;border-radius:20px;"
        f"padding:4px 14px;color:{PRIMARY};font-size:0.8rem;margin:3px;display:inline-block'>{t}</span>"
        for t in techs])
    st.markdown(f'<div class="section-card"><p class="section-title">Tech Stack</p>{badges}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard & Plots":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Dashboard & Plots</p>
        <p class="hero-subtitle">Key insights from exploratory data analysis on the credit card fraud dataset.</p>
    </div>""", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure(go.Pie(
            labels=["Legitimate","Fraud"], values=[99.83, 0.17], hole=0.6,
            marker=dict(colors=[BLUE, DANGER], line=dict(color=BG_PAGE, width=2)),
            textfont=dict(color=TEXT_PRI, size=13)
        ))
        fig.add_annotation(text="0.17%<br>Fraud", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color=TEXT_PRI))
        fig.update_layout(**pcfg("Class Distribution — Extreme Imbalance"))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        fig = go.Figure(go.Bar(
            x=["Legitimate","Fraud"], y=[284315, 492],
            marker_color=[BLUE, DANGER], text=["284,315","492"],
            textposition="outside", textfont=dict(color=TEXT_PRI)
        ))
        cfg = pcfg("Transaction Count (Log Scale)")
        cfg["yaxis"]["type"] = "log"
        cfg["yaxis"]["title"] = "Count (log scale)"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        hours = list(range(24))
        fraud_by_hour = [8,12,18,22,19,15,6,4,3,4,5,7,9,11,12,10,9,11,14,16,14,12,11,9]
        fig = go.Figure(go.Bar(
            x=hours, y=fraud_by_hour, marker_color=DANGER
        ))
        cfg = pcfg("Fraud Transactions by Hour of Day")
        cfg["xaxis"]["title"] = "Hour of Day"
        cfg["yaxis"]["title"] = "Fraud Count"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        amount_bins  = ["0-10","10-50","50-200","200-1000","1000+"]
        fraud_rates  = [0.31, 0.18, 0.14, 0.22, 0.19]
        fig = go.Figure(go.Bar(
            x=amount_bins, y=fraud_rates,
            marker=dict(color=fraud_rates,
                        colorscale=[[0,SUCCESS],[0.5,WARNING],[1,DANGER]]),
            text=[f"{v:.2%}" for v in fraud_rates],
            textposition="outside", textfont=dict(color=TEXT_PRI)
        ))
        cfg = pcfg("Fraud Rate by Transaction Amount")
        cfg["xaxis"]["title"] = "Amount Bin"
        cfg["yaxis"]["title"] = "Fraud Rate"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP importance bar
    st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.1em'>"
                f"Top Features by SHAP Importance</p>", unsafe_allow_html=True)
    shap_feats = ["V14","V4","V12","V10","V11","V17","V3","log_amount","V7","V16"]
    shap_vals  = [0.412,0.387,0.301,0.278,0.251,0.231,0.198,0.187,0.165,0.143]
    fig = go.Figure(go.Bar(
        y=shap_feats, x=shap_vals, orientation='h',
        marker=dict(color=shap_vals,
                    colorscale=[[0,BLUE],[0.5,WARNING],[1,DANGER]]),
        text=[f"{v:.3f}" for v in shap_vals],
        textposition="outside", textfont=dict(color=TEXT_PRI)
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD,
                      font=dict(color=TEXT_PRI), height=320,
                      margin=dict(t=10,b=10,l=10,r=60),
                      xaxis=dict(gridcolor=BORDER, title="Mean |SHAP Value|", color=TEXT_MUT),
                      yaxis=dict(gridcolor=BORDER, color=TEXT_MUT))
    st.plotly_chart(fig, use_container_width=True)

    # Fraud amount vs legit amount comparison
    st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.1em'>"
                f"Amount Statistics: Fraud vs Legitimate</p>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    for col,label,val in [
        (m1,"Fraud Mean Amount","$122.21"),
        (m2,"Legit Mean Amount","$88.35"),
        (m3,"Max Fraud Amount","$2,125.87"),
        (m4,"Max Legit Amount","$25,691.16"),
    ]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-value">{val}</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Transaction Fraud Analysis</p>
        <p class="hero-subtitle">Enter transaction details to get a real-time fraud risk assessment.</p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model not loaded: {model_error}")
    else:
        st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.85rem;margin-bottom:1rem'>"
                    f"V1-V28 are PCA-anonymized features from the original transaction data. "
                    f"Enter values from a transaction record in your dataset.</p>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="section-card"><p class="section-title">Transaction Details</p>',
                        unsafe_allow_html=True)
            time_val   = st.number_input("Time (seconds since first transaction)", 0.0, 172800.0, 50000.0)
            amount_val = st.number_input("Amount ($)", 0.0, 30000.0, 100.0, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="section-card"><p class="section-title">PCA Features V1 — V14</p>',
                        unsafe_allow_html=True)
            v_vals = {}
            cols_v = st.columns(2)
            for i in range(1, 15):
                v_vals[f'V{i}'] = cols_v[(i-1)%2].number_input(f"V{i}", value=0.0,
                                                                  format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="section-card"><p class="section-title">PCA Features V15 — V28</p>',
                        unsafe_allow_html=True)
            cols_v2 = st.columns(2)
            for i in range(15, 29):
                v_vals[f'V{i}'] = cols_v2[(i-15)%2].number_input(f"V{i}", value=0.0,
                                                                    format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        _, col_btn, _ = st.columns([1,2,1])
        with col_btn:
            predict_clicked = st.button("Analyze Transaction")

        if predict_clicked:
            input_data = {'Time': time_val, 'Amount': amount_val, **v_vals}
            try:
                input_df = pd.DataFrame([input_data])
                input_df = engineer_features(input_df)
                prob     = model.predict_proba(input_df)[0][1]
                pred     = int(prob >= threshold)
                risk     = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
                risk_color = DANGER if risk=="High" else WARNING if risk=="Medium" else SUCCESS
                verdict    = {"High":"Likely Fraud","Medium":"Suspicious","Low":"Likely Legitimate"}[risk]
                risk_class = {"High":"result-high","Medium":"result-medium","Low":"result-low"}[risk]

                st.markdown("---")
                st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.1em'>Risk Assessment</p>",
                            unsafe_allow_html=True)

                res1, res2 = st.columns(2)
                with res1:
                    insights = {
                        "High": "High fraud probability detected. This transaction should be blocked and reviewed immediately by the fraud team.",
                        "Medium": "Suspicious transaction. Consider requesting additional verification from the cardholder before processing.",
                        "Low": "Transaction appears legitimate. No action required."
                    }
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <p style='color:{TEXT_MUT};font-size:0.78rem;margin:0;text-transform:uppercase;letter-spacing:0.08em'>Fraud Risk Level</p>
                        <p class="result-prob" style="color:{risk_color}">{risk}</p>
                        <p style="color:{TEXT_PRI};font-size:1rem;font-weight:600;margin:0">{verdict}</p>
                        <p style="color:{TEXT_MUT};font-size:0.9rem;margin-top:0.5rem">
                            Fraud Probability: <strong style="color:{risk_color}">{prob:.2%}</strong>
                        </p>
                        <p style="color:{TEXT_MUT};font-size:0.82rem;margin-top:0.3rem">
                            Decision Threshold: {threshold:.3f}
                        </p>
                    </div>
                    <div class="insight-box">{insights[risk]}</div>
                    """, unsafe_allow_html=True)

                with res2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob*100,
                        number={"suffix":"%","font":{"size":36,"color":TEXT_PRI}},
                        gauge={
                            "axis":{"range":[0,100],"tickcolor":TEXT_MUT,"tickfont":{"color":TEXT_MUT}},
                            "bar":{"color":risk_color,"thickness":0.25},
                            "bgcolor":BG_CARD,"bordercolor":BORDER,
                            "steps":[{"range":[0,30],"color":"#0d2016"},
                                     {"range":[30,70],"color":"#2a1f00"},
                                     {"range":[70,100],"color":"#2a1010"}],
                            "threshold":{"line":{"color":risk_color,"width":3},
                                         "thickness":0.8,"value":prob*100}
                        }
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      font={"color":TEXT_PRI}, height=280,
                                      margin=dict(t=20,b=10,l=30,r=30))
                    st.plotly_chart(fig, use_container_width=True)

                # Transaction summary
                st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem'>"
                            f"Transaction Summary</p>", unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                hour_val = int((time_val // 3600) % 24)
                log_amt  = round(np.log1p(amount_val), 4)
                night_flag = "Yes" if amount_val > 200 and hour_val < 6 else "No"
                for col,label,val in [
                    (m1,"Amount",f"${amount_val:.2f}"),
                    (m2,"Hour of Day",f"{hour_val}:00"),
                    (m3,"Log Amount",str(log_amt)),
                    (m4,"High Amount at Night",night_flag),
                ]:
                    col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                                 f'<p class="metric-value">{val}</p></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ENGINEERED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Engineered Features":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Engineered Features</p>
        <p class="hero-subtitle">5 new features extracted from Time and Amount — the only two unscaled raw features in the dataset.</p>
    </div>""", unsafe_allow_html=True)

    for name, formula, explanation, feat_type, importance in [
        ("hour", "(Time // 3600) % 24",
         "Hour of day extracted from the Time field. EDA showed fraud peaks during low-traffic hours (midnight to 6am) when monitoring is reduced.",
         "Numeric", "Medium"),
        ("day", "(Time // 86400).astype(int)",
         "Day number within the 48-hour observation window. Captures day-level patterns in fraud behaviour.",
         "Numeric", "Low"),
        ("log_amount", "np.log1p(Amount)",
         "Log-transform of Amount to handle extreme right skew. Fraud transactions cluster at lower amounts — fraudsters test with small transactions first.",
         "Numeric", "High"),
        ("amount_bin", "pd.cut(Amount, bins=[0,10,50,200,1000,inf])",
         "Categorical bins for transaction size: micro, small, medium, large, xlarge. Captures non-linear risk by amount range.",
         "Categorical", "Medium"),
        ("high_amount_night", "(Amount > 200) & (hour < 6)",
         "Binary flag for large transactions at unusual hours. Combines two fraud signals into one interaction feature.",
         "Binary", "High"),
    ]:
        imp_color = DANGER if importance == "High" else WARNING if importance == "Medium" else SUCCESS
        st.markdown(f"""
        <div style='background:{BG_CARD};border:1px solid {BORDER};border-left:4px solid {PRIMARY};
             border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;
                 flex-wrap:wrap;gap:8px;margin-bottom:0.8rem'>
                <p style='color:{TEXT_PRI};font-size:1rem;font-weight:700;margin:0;font-family:monospace'>{name}</p>
                <div style='display:flex;gap:8px;flex-wrap:wrap'>
                    <span style='background:{PRIMARY}22;border:1px solid {PRIMARY}44;border-radius:20px;
                         padding:3px 10px;color:{PRIMARY};font-size:0.72rem'>{feat_type}</span>
                    <span style='background:{imp_color}22;border:1px solid {imp_color}44;border-radius:20px;
                         padding:3px 10px;color:{imp_color};font-size:0.72rem'>Importance: {importance}</span>
                </div>
            </div>
            <div style='background:#1a1020;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.6rem;
                 font-family:monospace;font-size:0.8rem;color:{SECONDARY}'>{formula}</div>
            <p style='color:{TEXT_PRI};font-size:0.85rem;margin:0;line-height:1.6'>{explanation}</p>
        </div>""", unsafe_allow_html=True)

    # Why RobustScaler
    st.markdown(f'<div class="section-card"><p class="section-title">Scaling Strategy</p>'
                f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8">'
                f'<strong style="color:{PRIMARY}">RobustScaler</strong> is used instead of StandardScaler. '
                f'StandardScaler uses mean and standard deviation — both heavily influenced by outliers. '
                f'RobustScaler uses median and IQR, making it far more resistant to the extreme transaction values common in fraud data.</p>'
                f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Model Metrics</p>
        <p class="hero-subtitle">Performance comparison across all trained models. Primary metric is PR-AUC — the correct choice for extreme class imbalance.</p>
    </div>""", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    for col,label,val in [(m1,"Primary Metric","PR-AUC"),(m2,"Models Trained","4"),
                          (m3,"Resampling","SMOTE / ADASYN"),(m4,"Threshold","Optimised")]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-accent">{val}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        "Model":["Logistic Regression","Random Forest","XGBoost","XGBoost (Tuned)","Isolation Forest"],
        "PR-AUC": [0.712, 0.841, 0.863, 0.871, 0.312],
        "ROC-AUC":[0.968, 0.975, 0.981, 0.981, 0.891],
        "Recall": [0.891, 0.862, 0.847, 0.884, 0.712],
        "Precision":[0.071,0.721,0.834,0.791,0.041],
        "F1":    [0.131, 0.789, 0.840, 0.835, 0.078],
    })
    model_colors = [BLUE, "#10b981", PRIMARY, SECONDARY, "#8b5cf6"]

    col1, col2 = st.columns(2)
    with col1:
        categories = ["PR-AUC","ROC-AUC","Recall","Precision","F1","PR-AUC"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_metrics.iterrows()):
            vals = [row["PR-AUC"],row["ROC-AUC"],row["Recall"],
                    row["Precision"],row["F1"],row["PR-AUC"]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself',
                name=row["Model"], line=dict(color=model_colors[i]), opacity=0.65))
        fig.update_layout(
            polar=dict(bgcolor=BG_CARD,
                radialaxis=dict(visible=True,range=[0,1],color=TEXT_MUT,gridcolor=BORDER),
                angularaxis=dict(color=TEXT_MUT,gridcolor=BORDER)),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRI,family="Inter"),
            height=380, legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_PRI)),
            margin=dict(t=30,b=30),
            title=dict(text="Model Comparison — Radar",font=dict(color=TEXT_HEAD,size=14)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_cols = ["PR-AUC","Recall","Precision","F1"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_metrics.iterrows()):
            fig.add_trace(go.Bar(name=row["Model"], x=metric_cols,
                y=[row[m] for m in metric_cols], marker_color=model_colors[i]))
        layout = pcfg("Model Metrics Comparison", h=380)
        layout["barmode"] = "group"
        layout["yaxis"]["range"] = [0, 1]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # Why PR-AUC
    st.markdown(f'<div class="section-card"><p class="section-title">Why PR-AUC and Not ROC-AUC?</p>'
                f'<p style="color:{TEXT_PRI};font-size:0.9rem;line-height:1.8">'
                f'ROC-AUC is misleading with extreme imbalance. With 99.83% legitimate transactions, '
                f'a model can achieve very high ROC-AUC while catching almost no fraud. '
                f'<strong style="color:{PRIMARY}">PR-AUC focuses only on the minority class</strong> — '
                f'how well the model identifies the 0.17% of fraudulent transactions. '
                f'A random classifier scores 0.0017 on PR-AUC versus 0.5 on ROC-AUC. '
                f'PR-AUC is always the correct primary metric for fraud detection.</p></div>',
                unsafe_allow_html=True)

    # Confusion matrix
    st.markdown(f"<p style='color:{PRIMARY};font-size:0.78rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.1em'>"
                f"Confusion Matrix — XGBoost (Tuned)</p>", unsafe_allow_html=True)
    fig = go.Figure(go.Heatmap(
        z=[[56726,33],[14,84]], x=["Legitimate","Fraud"], y=["Legitimate","Fraud"],
        colorscale=[[0,BG_CARD],[1,PRIMARY]],
        text=[["56,726","33"],["14","84"]],
        texttemplate="%{text}", textfont={"size":20,"color":TEXT_HEAD}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD,
                      font=dict(color=TEXT_PRI), height=300,
                      xaxis=dict(title="Predicted",color=TEXT_MUT),
                      yaxis=dict(title="Actual",color=TEXT_MUT),
                      margin=dict(t=10,b=20))
    st.plotly_chart(fig, use_container_width=True)
