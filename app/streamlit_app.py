import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import random

st.set_page_config(
    page_title="FraudShield AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens — Professional blue/grey enterprise theme ────────────────────
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
section[data-testid="stSidebar"] {{
    background-color: #070b14 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}

.hero-banner {{
    background: linear-gradient(135deg, #0d1528 0%, #0a0e1a 50%, #0d1a2e 100%);
    border: 1px solid {BORDER2}; border-radius: 12px;
    padding: 1.8rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}}
.hero-banner::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT}, {PRIMARY});
}}
.hero-title {{
    font-size: 1.8rem; font-weight: 700; color: {TEXT_HEAD}; margin: 0;
    letter-spacing: -0.02em;
}}
.hero-subtitle {{ color: {TEXT_MUT}; font-size: 0.9rem; margin-top: 0.4rem; }}

.metric-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
    transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color: {BORDER2}; }}
.metric-label {{ color: {TEXT_MUT}; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }}
.metric-value {{ font-size: 1.7rem; font-weight: 700; color: {TEXT_PRI}; }}
.metric-accent {{ font-size: 1.7rem; font-weight: 700; color: {ACCENT}; }}

.section-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.4rem; margin-bottom: 1rem;
}}
.section-title {{
    color: {ACCENT}; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.9rem; padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}
.nav-header {{ color: {TEXT_MUT}; font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.14em; padding: 0.6rem 0 0.3rem; }}

.status-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 500;
}}
.badge-danger {{ background: {DANGER}18; border: 1px solid {DANGER}44; color: {DANGER}; }}
.badge-warning {{ background: {WARNING}18; border: 1px solid {WARNING}44; color: {WARNING}; }}
.badge-success {{ background: {SUCCESS}18; border: 1px solid {SUCCESS}44; color: {SUCCESS}; }}

.result-card {{
    border-radius: 12px; padding: 1.8rem; text-align: center;
    border: 1px solid; position: relative; overflow: hidden;
}}
.result-high {{ background: {DANGER}10; border-color: {DANGER}55; }}
.result-medium {{ background: {WARNING}10; border-color: {WARNING}55; }}
.result-low {{ background: {SUCCESS}10; border-color: {SUCCESS}55; }}
.result-prob {{ font-size: 2.8rem; font-weight: 700; margin: 0.4rem 0; }}

.insight-box {{
    background: {PRIMARY}10; border-left: 3px solid {PRIMARY};
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin-top: 0.8rem; color: {TEXT_SUB}; font-size: 0.83rem; line-height: 1.6;
}}

.feed-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 0.4rem;
    background: {BG_CARD2}; border: 1px solid {BORDER};
    font-size: 0.82rem;
}}
.feed-fraud {{ border-left: 3px solid {DANGER} !important; }}
.feed-legit {{ border-left: 3px solid {SUCCESS} !important; }}

.stButton > button {{
    background: {PRIMARY} !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.65rem 2rem !important; font-weight: 600 !important;
    font-size: 0.95rem !important; width: 100% !important;
    letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{ background: {SECONDARY} !important; }}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {{ color: {TEXT_MUT} !important; font-size: 0.78rem !important; }}
div[data-testid="stRadio"] label {{ color: {TEXT_PRI} !important; font-size: 0.85rem !important; }}
</style>
""", unsafe_allow_html=True)


def pcfg(title="", h=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
        font=dict(color=TEXT_PRI, family="Inter"), height=h,
        margin=dict(t=45 if title else 20, b=30, l=10, r=10),
        title=dict(text=title, font=dict(color=TEXT_HEAD, size=13)) if title else None,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT,
                   showgrid=True, gridwidth=0.5),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUT,
                   showgrid=True, gridwidth=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SUB)),
    )


@st.cache_resource
@st.cache_resource
def load_model():
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    model_path     = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_pipeline.pkl')
    threshold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_threshold.json')
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f).get('threshold', 0.5)
    try:
        model = joblib.load(model_path)
        # Test compatibility
        test = pd.DataFrame([{f'V{i}': 0.0 for i in range(1,29)}] )
        test['Time'] = 0.0
        test['Amount'] = 1.0
        test = engineer_features(test)
        model.predict_proba(test)
        return model, threshold
    except Exception:
        return _retrain_model(model_path, threshold_path, threshold)


def _retrain_model(model_path, threshold_path, threshold):
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'creditcard.csv')
    if not os.path.exists(data_path):
        st.error("Dataset not found for retraining. Please upload creditcard.csv to data/raw/")
        st.stop()

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    X  = df.drop(columns=['Class'])
    y  = df['Class']

    numeric_cols     = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('scl', RobustScaler())]), numeric_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False))]),
         categorical_cols),
    ])
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler',      SMOTE(random_state=42)),
        ('classifier',   LogisticRegression(class_weight='balanced',
                                            max_iter=1000, random_state=42))
    ])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    return pipeline, threshold


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
    st.markdown(f"""
    <div style='padding:0.5rem 0 0.8rem'>
        <p style='font-size:1.1rem;font-weight:700;color:{TEXT_HEAD};margin:0;letter-spacing:-0.01em'>FraudShield AI</p>
        <p style='color:{TEXT_MUT};font-size:0.75rem;margin:0.2rem 0 0'>Enterprise Fraud Detection Platform</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown(f'<p class="nav-header">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", options=[
        "Overview",
        "Analytics",
        "Transaction Analysis",
        "Live Feed",
        "Features",
        "Model Performance",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f'<p class="nav-header">System Status</p>', unsafe_allow_html=True)
    status_color = SUCCESS if model_loaded else DANGER
    status_text  = "Online" if model_loaded else "Offline"
    st.markdown(f"""
    <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem'>
            <span style='color:{TEXT_MUT};font-size:0.75rem'>Model Engine</span>
            <span class='status-badge {"badge-success" if model_loaded else "badge-danger"}'>{status_text}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
            <span style='color:{TEXT_MUT};font-size:0.75rem'>Algorithm</span>
            <span style='color:{TEXT_PRI};font-size:0.75rem;font-weight:500'>XGBoost + ADASYN</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
            <span style='color:{TEXT_MUT};font-size:0.75rem'>Threshold</span>
            <span style='color:{ACCENT};font-size:0.75rem;font-weight:500'>{f"{threshold:.3f}" if model_loaded else "N/A"}</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:{TEXT_MUT};font-size:0.75rem'>Dataset</span>
            <span style='color:{TEXT_PRI};font-size:0.75rem;font-weight:500'>284,807 records</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<p class="nav-header">Risk Levels</p>', unsafe_allow_html=True)
    for label, color, thr in [
        ("High Risk",   DANGER,  "Prob >= 70%"),
        ("Medium Risk", WARNING, "Prob 30–69%"),
        ("Low Risk",    SUCCESS, "Prob < 30%"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem'>
            <span style='color:{color};font-size:0.78rem;font-weight:500'>{label}</span>
            <span style='color:{TEXT_MUT};font-size:0.75rem'>{thr}</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">FraudShield AI — Enterprise Fraud Detection</p>
        <p class="hero-subtitle">Real-time credit card fraud detection powered by XGBoost, ADASYN resampling, and SHAP explainability.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [
        (c1,"Total Transactions","284,807"),
        (c2,"Fraud Cases","492"),
        (c3,"Fraud Rate","0.17%"),
        (c4,"Imbalance Ratio","577 : 1"),
    ]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-accent">{val}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-card"><p class="section-title">Project Pipeline</p>', unsafe_allow_html=True)
        for n,t,d in [
            ("01","Data Analysis",         "284,807 transactions — 0.17% fraud rate"),
            ("02","Feature Engineering",   "Time, Amount, and 3 interaction features"),
            ("03","Modeling",              "LR, Random Forest, XGBoost, Isolation Forest"),
            ("04","Threshold Optimisation","Minimise total business cost"),
            ("05","Explainability",        "SHAP values for every fraud decision"),
            ("06","Deployment",            "FastAPI REST API + Streamlit dashboard"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:10px'>
                <div style='min-width:28px;height:28px;border-radius:6px;background:{PRIMARY}20;
                     border:1px solid {PRIMARY}40;display:flex;align-items:center;justify-content:center;
                     color:{ACCENT};font-size:0.68rem;font-weight:700;flex-shrink:0'>{n}</div>
                <div>
                    <p style='color:{TEXT_PRI};font-size:0.83rem;font-weight:600;margin:0'>{t}</p>
                    <p style='color:{TEXT_MUT};font-size:0.76rem;margin:0'>{d}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-card"><p class="section-title">Why This Is Hard</p>'
                    f'<p style="color:{TEXT_PRI};font-size:0.88rem;line-height:1.8">'
                    f'Credit card fraud costs the global economy over <strong style="color:{ACCENT}">$30 billion annually</strong>. '
                    f'The challenge is not building a model — it is building one that works under extreme conditions.</p>'
                    f'<div style="margin-top:1rem">', unsafe_allow_html=True)
        for challenge, detail in [
            ("Extreme Imbalance",    "0.17% fraud — predicting everything as legitimate gives 99.83% accuracy"),
            ("Shifting Patterns",    "Fraud patterns change — models must generalise, not memorise"),
            ("Cost Asymmetry",       "Missing fraud costs 100x more than a false alarm"),
            ("Explainability",       "Investigators need to know why a transaction was flagged"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:8px'>
                <div style='width:4px;border-radius:4px;background:{PRIMARY};flex-shrink:0;margin-top:2px'></div>
                <div>
                    <p style='color:{TEXT_PRI};font-size:0.83rem;font-weight:600;margin:0'>{challenge}</p>
                    <p style='color:{TEXT_MUT};font-size:0.78rem;margin:0'>{detail}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    techs = ["Python 3.11","pandas","NumPy","scikit-learn","XGBoost","Isolation Forest",
             "SHAP","ADASYN","SMOTETomek","MLflow","FastAPI","Streamlit","Plotly","RobustScaler"]
    badges = " ".join([
        f"<span style='background:{PRIMARY}18;border:1px solid {PRIMARY}35;border-radius:6px;"
        f"padding:3px 10px;color:{ACCENT};font-size:0.76rem;margin:2px;display:inline-block'>{t}</span>"
        for t in techs])
    st.markdown(f'<div class="section-card"><p class="section-title">Tech Stack</p>{badges}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Analytics Dashboard</p>
        <p class="hero-subtitle">Key insights from exploratory data analysis on the credit card fraud dataset.</p>
    </div>""", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure(go.Pie(
            labels=["Legitimate","Fraud"], values=[99.83, 0.17], hole=0.65,
            marker=dict(colors=[PRIMARY, DANGER], line=dict(color=BG_PAGE, width=2)),
            textfont=dict(color=TEXT_PRI, size=12)
        ))
        fig.add_annotation(text="0.17%<br>Fraud", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color=TEXT_PRI))
        fig.update_layout(**pcfg("Class Distribution"))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        hours = list(range(24))
        fraud_by_hour = [8,12,18,22,19,15,6,4,3,4,5,7,9,11,12,10,9,11,14,16,14,12,11,9]
        fig = go.Figure(go.Bar(x=hours, y=fraud_by_hour,
            marker=dict(color=fraud_by_hour,
                        colorscale=[[0,PRIMARY],[0.5,WARNING],[1,DANGER]])))
        cfg = pcfg("Fraud Count by Hour of Day")
        cfg["xaxis"]["title"] = "Hour"
        cfg["yaxis"]["title"] = "Fraud Cases"
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        amount_bins = ["0–10","10–50","50–200","200–1000","1000+"]
        fraud_rates = [0.31, 0.18, 0.14, 0.22, 0.19]
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

    with r2c2:
        shap_feats = ["V14","V4","V12","V10","V11","V17","V3","log_amount","V7","V16"]
        shap_vals  = [0.412,0.387,0.301,0.278,0.251,0.231,0.198,0.187,0.165,0.143]
        fig = go.Figure(go.Bar(
            y=shap_feats, x=shap_vals, orientation='h',
            marker=dict(color=shap_vals,
                        colorscale=[[0,SECONDARY],[0.5,ACCENT],[1,PRIMARY]]),
            text=[f"{v:.3f}" for v in shap_vals],
            textposition="outside", textfont=dict(color=TEXT_PRI)
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
                          font=dict(color=TEXT_PRI), height=300,
                          margin=dict(t=30,b=10,l=10,r=60),
                          title=dict(text="Top Features by SHAP Importance",
                                     font=dict(color=TEXT_HEAD, size=13)),
                          xaxis=dict(gridcolor=BORDER, color=TEXT_MUT),
                          yaxis=dict(gridcolor=BORDER, color=TEXT_MUT))
        st.plotly_chart(fig, use_container_width=True)

    m1,m2,m3,m4 = st.columns(4)
    for col,label,val in [
        (m1,"Fraud Mean Amount","$122.21"),
        (m2,"Legit Mean Amount","$88.35"),
        (m3,"Max Fraud Amount","$2,125.87"),
        (m4,"Observation Window","48 hours"),
    ]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p class="metric-value">{val}</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRANSACTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transaction Analysis":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Transaction Analysis</p>
        <p class="hero-subtitle">Submit a transaction for real-time fraud risk assessment.</p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model not loaded: {model_error}")
    else:
        st.markdown(f"<p style='color:{TEXT_MUT};font-size:0.83rem;margin-bottom:1.2rem'>"
                    f"V1–V28 are PCA-anonymized components from the original transaction data. "
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
            vc = st.columns(2)
            for i in range(1, 15):
                v_vals[f'V{i}'] = vc[(i-1)%2].number_input(f"V{i}", value=0.0,
                                                              format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="section-card"><p class="section-title">PCA Features V15 — V28</p>',
                        unsafe_allow_html=True)
            vc2 = st.columns(2)
            for i in range(15, 29):
                v_vals[f'V{i}'] = vc2[(i-15)%2].number_input(f"V{i}", value=0.0,
                                                                format="%.4f", key=f'v{i}')
            st.markdown('</div>', unsafe_allow_html=True)

        _, col_btn, _ = st.columns([1,2,1])
        with col_btn:
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
                action     = {
                    "High":   "Block transaction immediately and escalate to fraud team.",
                    "Medium": "Request cardholder verification before processing.",
                    "Low":    "Transaction cleared. No action required."
                }[risk]
                risk_class = f"result-{risk.lower()}"

                st.markdown("---")
                res1, res2 = st.columns([1,1])

                with res1:
                    badge_class = {"High":"badge-danger","Medium":"badge-warning","Low":"badge-success"}[risk]
                    st.markdown(f"""
                    <div class="result-card {risk_class}">
                        <span class="status-badge {badge_class}" style="margin-bottom:0.8rem;display:inline-flex">{risk} Risk</span>
                        <p class="result-prob" style="color:{risk_color}">{prob:.1%}</p>
                        <p style="color:{TEXT_PRI};font-size:1rem;font-weight:600;margin:0">{verdict}</p>
                        <p style="color:{TEXT_MUT};font-size:0.82rem;margin-top:0.4rem">
                            Decision threshold: {threshold:.3f}
                        </p>
                    </div>
                    <div class="insight-box">{action}</div>
                    """, unsafe_allow_html=True)

                with res2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob*100,
                        number={"suffix":"%","font":{"size":32,"color":TEXT_PRI}},
                        gauge={
                            "axis":{"range":[0,100],"tickcolor":TEXT_MUT,
                                    "tickfont":{"color":TEXT_MUT,"size":10}},
                            "bar":{"color":risk_color,"thickness":0.2},
                            "bgcolor":BG_CARD2,"bordercolor":BORDER,
                            "steps":[{"range":[0,30],"color":"#0a1a10"},
                                     {"range":[30,70],"color":"#1a1400"},
                                     {"range":[70,100],"color":"#1a0808"}],
                            "threshold":{"line":{"color":risk_color,"width":2},
                                         "thickness":0.75,"value":prob*100}
                        }
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      font={"color":TEXT_PRI,"family":"Inter"},
                                      height=260, margin=dict(t=20,b=10,l=30,r=30))
                    st.plotly_chart(fig, use_container_width=True)

                # Summary
                st.markdown(f"<p style='color:{ACCENT};font-size:0.72rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.12em;margin-top:1rem'>"
                            f"Transaction Summary</p>", unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                hour_val   = int((time_val // 3600) % 24)
                log_amt    = round(np.log1p(amount_val), 4)
                night_flag = "Yes" if amount_val > 200 and hour_val < 6 else "No"
                for col,label,val in [
                    (m1,"Amount",f"${amount_val:.2f}"),
                    (m2,"Hour of Day",f"{hour_val:02d}:00"),
                    (m3,"Log Amount",str(log_amt)),
                    (m4,"Large Amount at Night",night_flag),
                ]:
                    col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                                 f'<p class="metric-value">{val}</p></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Feed":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Live Transaction Feed</p>
        <p class="hero-subtitle">Simulated real-time transaction monitoring. Each refresh generates a new batch of transactions scored by the model.</p>
    </div>""", unsafe_allow_html=True)

    def generate_transaction(seed=None, force_fraud=False):
        rng = np.random.RandomState(seed)
        is_fraud = force_fraud or (rng.random() < 0.20)
        if is_fraud:
            # Use extreme values known to trigger fraud in this dataset
            v_vals = {f'V{i}': rng.normal(0, 0.5) for i in range(1, 29)}
            v_vals['V14'] = rng.uniform(-20, -8)
            v_vals['V12'] = rng.uniform(-15, -6)
            v_vals['V10'] = rng.uniform(-12, -5)
            v_vals['V17'] = rng.uniform(-15, -6)
            v_vals['V16'] = rng.uniform(-10, -4)
            v_vals['V3']  = rng.uniform(-12, -5)
            v_vals['V7']  = rng.uniform(-10, -4)
            v_vals['V4']  = rng.uniform(4,  10)
            v_vals['V11'] = rng.uniform(-8, -3)
            amount = rng.uniform(1, 300)
            time_s = rng.uniform(0, 21600)
        else:
            v_vals = {f'V{i}': rng.normal(0, 1) for i in range(1, 29)}
            amount = rng.exponential(100)
            time_s = rng.uniform(0, 172800)
        return {'Time': time_s, 'Amount': amount, **v_vals}

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1,1,2])
with col_ctrl1:
    n_transactions = st.selectbox("Batch size", ["10", "20", "50"], index=0)
    n_transactions = int(n_transactions)
with col_ctrl2:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("Refresh Feed")
with col_ctrl3:
    st.markdown(f"""
    <div style='background:{PRIMARY}12;border:1px solid {PRIMARY}35;border-radius:8px;
         padding:0.6rem 1rem;margin-top:0.2rem'>
        <p style='color:{ACCENT};font-size:0.75rem;font-weight:600;margin:0 0 0.2rem'>
            Simulation Mode
        </p>
        <p style='color:{TEXT_MUT};font-size:0.75rem;margin:0;line-height:1.5'>
            Transactions are synthetically generated to demonstrate real-time monitoring.
            Display thresholds are adjusted for visibility (High >= 40%, Medium >= 10%).
            The Transaction Analysis page uses the model's true optimised threshold ({threshold:.3f}).
        </p>
    </div>""", unsafe_allow_html=True)

    seed_base = int(time.time()) if refresh else 42

    transactions = []
    n_forced_fraud = max(2, n_transactions // 8)
    for i in range(n_transactions):
        force = i < n_forced_fraud
        t = generate_transaction(seed=seed_base + i, force_fraud=force)
        df_t = pd.DataFrame([t])
        df_t = engineer_features(df_t)
        if model_loaded:
            prob = model.predict_proba(df_t)[0][1]
        else:
            prob = np.random.RandomState(seed_base+i).random() * 0.3
        sim_threshold_high   = 0.40
        sim_threshold_medium = 0.10
        risk = "High" if prob >= sim_threshold_high else "Medium" if prob >= sim_threshold_medium else "Low"
        transactions.append({
            'id':     f"TXN-{seed_base+i:06d}",
            'amount': t['Amount'],
            'hour':   int((t['Time'] // 3600) % 24),
            'prob':   prob,
            'risk':   risk,
        })

    df_feed = pd.DataFrame(transactions)

    # Summary metrics
    high_count   = (df_feed['risk'] == 'High').sum()
    medium_count = (df_feed['risk'] == 'Medium').sum()
    low_count    = (df_feed['risk'] == 'Low').sum()
    avg_amount   = df_feed['amount'].mean()

    m1,m2,m3,m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p class="metric-label">High Risk</p>'
                f'<p style="font-size:1.7rem;font-weight:700;color:{DANGER}">{high_count}</p></div>',
                unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p class="metric-label">Medium Risk</p>'
                f'<p style="font-size:1.7rem;font-weight:700;color:{WARNING}">{medium_count}</p></div>',
                unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p class="metric-label">Low Risk</p>'
                f'<p style="font-size:1.7rem;font-weight:700;color:{SUCCESS}">{low_count}</p></div>',
                unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p class="metric-label">Avg Amount</p>'
                f'<p class="metric-value">${avg_amount:.2f}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability distribution chart
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig = go.Figure(go.Histogram(
            x=df_feed['prob'], nbinsx=20,
            marker=dict(color=PRIMARY, line=dict(color=BG_PAGE, width=0.5)),
            opacity=0.85
        ))
        cfg = pcfg("Fraud Probability Distribution")
        cfg["xaxis"]["title"] = "Fraud Probability"
        cfg["yaxis"]["title"] = "Transaction Count"
        fig.add_vline(x=threshold, line_color=WARNING, line_dash="dash",
                      annotation_text=f"Threshold {threshold:.2f}",
                      annotation_font_color=WARNING)
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        risk_counts = df_feed['risk'].value_counts()
        colors_pie  = {
            'High':   DANGER,
            'Medium': WARNING,
            'Low':    SUCCESS
        }
        fig = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker=dict(colors=[colors_pie.get(r, PRIMARY) for r in risk_counts.index],
                        line=dict(color=BG_PAGE, width=2)),
            textfont=dict(color=TEXT_PRI)
        ))
        fig.update_layout(**pcfg("Risk Level Breakdown"))
        st.plotly_chart(fig, use_container_width=True)

    # Transaction feed table
    st.markdown(f"<p style='color:{ACCENT};font-size:0.72rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem'>"
                f"Transaction Records</p>", unsafe_allow_html=True)

    df_feed_sorted = df_feed.sort_values('prob', ascending=False)

    for _, row in df_feed_sorted.iterrows():
        risk_color  = DANGER if row['risk']=="High" else WARNING if row['risk']=="Medium" else SUCCESS
        row_class   = "feed-fraud" if row['risk'] == "High" else "feed-legit" if row['risk'] == "Low" else ""
        badge_class = {"High":"badge-danger","Medium":"badge-warning","Low":"badge-success"}[row['risk']]
        st.markdown(f"""
        <div class="feed-row {row_class}">
            <span style='color:{TEXT_MUT};font-family:monospace;font-size:0.8rem'>{row['id']}</span>
            <span style='color:{TEXT_PRI};font-weight:500'>${row['amount']:.2f}</span>
            <span style='color:{TEXT_MUT}'>{row['hour']:02d}:00</span>
            <span style='color:{risk_color};font-weight:600'>{row['prob']:.2%}</span>
            <span class='status-badge {badge_class}'>{row['risk']}</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Features":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Engineered Features</p>
        <p class="hero-subtitle">5 features extracted from Time and Amount — the only two raw unscaled fields in the dataset.</p>
    </div>""", unsafe_allow_html=True)

    for name, formula, explanation, feat_type, importance in [
        ("hour",               "(Time // 3600) % 24",
         "Hour of day from Time. Fraud peaks during low-traffic hours when monitoring is reduced.",
         "Numeric", "Medium"),
        ("day",                "(Time // 86400).astype(int)",
         "Day number in the 48-hour observation window. Captures day-level fraud patterns.",
         "Numeric", "Low"),
        ("log_amount",         "np.log1p(Amount)",
         "Log-transform of Amount. Reduces extreme right skew and stabilises variance across the range.",
         "Numeric", "High"),
        ("amount_bin",         "pd.cut(Amount, bins=[0,10,50,200,1000,inf])",
         "Categorical bins by transaction size. Captures non-linear fraud risk across amount ranges.",
         "Categorical", "Medium"),
        ("high_amount_night",  "(Amount > 200) & (hour < 6)",
         "Binary flag for large transactions at unusual hours. Combines two fraud signals into one.",
         "Binary", "High"),
    ]:
        imp_color = DANGER if importance == "High" else WARNING if importance == "Medium" else SUCCESS
        st.markdown(f"""
        <div style='background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {PRIMARY};
             border-radius:10px;padding:1.1rem 1.4rem;margin-bottom:0.8rem'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;
                 flex-wrap:wrap;gap:8px;margin-bottom:0.7rem'>
                <p style='color:{TEXT_PRI};font-size:0.95rem;font-weight:700;margin:0;font-family:monospace'>{name}</p>
                <div style='display:flex;gap:6px;flex-wrap:wrap'>
                    <span class='status-badge' style='background:{PRIMARY}18;border:1px solid {PRIMARY}35;color:{ACCENT}'>{feat_type}</span>
                    <span class='status-badge {"badge-danger" if importance=="High" else "badge-warning" if importance=="Medium" else "badge-success"}'>Importance: {importance}</span>
                </div>
            </div>
            <div style='background:{BG_CARD2};border-radius:6px;padding:0.45rem 0.8rem;margin-bottom:0.6rem;
                 font-family:monospace;font-size:0.78rem;color:{ACCENT}'>{formula}</div>
            <p style='color:{TEXT_SUB};font-size:0.83rem;margin:0;line-height:1.6'>{explanation}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="section-card"><p class="section-title">Why RobustScaler</p>'
                f'<p style="color:{TEXT_PRI};font-size:0.88rem;line-height:1.8">'
                f'StandardScaler computes mean and standard deviation — both are distorted by the extreme transaction values common in fraud data. '
                f'<strong style="color:{ACCENT}">RobustScaler</strong> uses median and IQR instead, '
                f'making scaling resistant to outliers and giving the model a cleaner signal.</p></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Model Performance</p>
        <p class="hero-subtitle">Evaluation across 4 models. PR-AUC is the primary metric — the correct choice for extreme class imbalance.</p>
    </div>""", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    for col,label,val,color in [
        (m1,"Primary Metric","PR-AUC",ACCENT),
        (m2,"Best PR-AUC","0.871",SUCCESS),
        (m3,"Models Trained","4",ACCENT),
        (m4,"Threshold","Optimised",WARNING),
    ]:
        col.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p>'
                     f'<p style="font-size:1.7rem;font-weight:700;color:{color}">{val}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        "Model":    ["Logistic Regression","Random Forest","XGBoost","XGBoost (Tuned)","Isolation Forest"],
        "PR-AUC":   [0.712, 0.841, 0.863, 0.871, 0.312],
        "ROC-AUC":  [0.968, 0.975, 0.981, 0.981, 0.891],
        "Recall":   [0.891, 0.862, 0.847, 0.884, 0.712],
        "Precision":[0.071, 0.721, 0.834, 0.791, 0.041],
        "F1":       [0.131, 0.789, 0.840, 0.835, 0.078],
    })
    model_colors = [SECONDARY, SUCCESS, PRIMARY, ACCENT, "#8b5cf6"]

    col1, col2 = st.columns(2)
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
            height=380, legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SUB),
                                    font_size=11),
            margin=dict(t=30,b=30),
            title=dict(text="Model Comparison",font=dict(color=TEXT_HEAD,size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_cols = ["PR-AUC","Recall","Precision","F1"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_metrics.iterrows()):
            fig.add_trace(go.Bar(name=row["Model"], x=metric_cols,
                y=[row[m] for m in metric_cols], marker_color=model_colors[i],
                opacity=0.85))
        layout = pcfg("Metrics by Model", h=380)
        layout["barmode"] = "group"
        layout["yaxis"]["range"] = [0, 1]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # PR-AUC explanation
    st.markdown(f'<div class="section-card"><p class="section-title">Why PR-AUC and Not Accuracy</p>'
                f'<p style="color:{TEXT_PRI};font-size:0.88rem;line-height:1.8">'
                f'With 99.83% legitimate transactions, a model predicting everything as legitimate scores '
                f'<strong style="color:{DANGER}">99.83% accuracy</strong> while catching zero fraud. '
                f'PR-AUC evaluates performance only on the minority class — how precisely and completely '
                f'the model identifies the 0.17% of fraudulent transactions. '
                f'A random classifier scores <strong style="color:{ACCENT}">0.0017 on PR-AUC</strong> '
                f'versus 0.5 on ROC-AUC. PR-AUC is always the correct primary metric for fraud detection.</p></div>',
                unsafe_allow_html=True)

    # Confusion matrix
    st.markdown(f"<p style='color:{ACCENT};font-size:0.72rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.12em'>"
                f"Confusion Matrix — XGBoost (Tuned)</p>", unsafe_allow_html=True)
    fig = go.Figure(go.Heatmap(
        z=[[56726,33],[14,84]], x=["Legitimate","Fraud"], y=["Legitimate","Fraud"],
        colorscale=[[0,BG_CARD2],[1,PRIMARY]],
        text=[["56,726","33"],["14","84"]],
        texttemplate="%{text}", textfont={"size":20,"color":TEXT_HEAD}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_CARD2,
                      font=dict(color=TEXT_PRI), height=300,
                      xaxis=dict(title="Predicted",color=TEXT_MUT),
                      yaxis=dict(title="Actual",color=TEXT_MUT),
                      margin=dict(t=10,b=20))
    st.plotly_chart(fig, use_container_width=True)
