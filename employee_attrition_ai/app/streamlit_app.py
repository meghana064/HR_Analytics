"""
AI Workforce Guardian - Employee Attrition Predictor
Streamlit dashboard for HR analytics and attrition risk prediction.
Netflix-style dark theme UI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import load_data, preprocess_data

# Page configuration - wide layout
st.set_page_config(
    page_title="AI Workforce Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix-style dark theme CSS
st.markdown("""
<style>
    /* Dark theme - Netflix/SaaS style */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 100%;
    }
    
    /* Premium header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 20px rgba(229, 9, 20, 0.3);
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* KPI cards - dark, modern, hover */
    .kpi-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.2);
        border-color: #e50914;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
    }
    
    /* Risk level styling */
    .risk-low { color: #22c55e; font-weight: bold; }
    .risk-medium { color: #eab308; font-weight: bold; }
    .risk-high { color: #e50914; font-weight: bold; }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9;
    }
    
    /* Sidebar nav - clean text links, no radio/circles/icons */
    [data-testid="stSidebar"] button {
        background: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        width: 100% !important;
        padding: 0.5rem 0 !important;
        margin-bottom: 0.25rem !important;
        transition: all 0.3s ease !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        color: #00BFFF !important;
        border-bottom: 2px solid #00BFFF !important;
    }
    
    /* AI Recommendations container */
    .ai-recommendations {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        border-left: 4px solid #e50914;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
    }
    .ai-recommendations h4 { color: #f1f5f9; margin-bottom: 1rem; }
    .ai-recommendations ul { color: #94a3b8; margin-left: 1.25rem; }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Main content text on dark */
    .stMarkdown, .stMarkdown p { color: #f1f5f9 !important; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

def load_dataset():
    """Load the HR dataset."""
    data_path = project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    return load_data(str(data_path))


def load_model_artifacts():
    """Load trained model and artifacts. Returns None if model not trained."""
    model_path = project_root / "model" / "attrition_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_employees_at_risk(df, artifacts):
    """Count employees at medium/high attrition risk (>= 30% probability)."""
    if artifacts is None:
        return None
    try:
        X, _, _, _ = preprocess_data(df)
        proba = artifacts["model"].predict_proba(X)[:, 1]
        return int((proba >= 0.3).sum())
    except Exception:
        return None


def get_at_risk_employees_df(df, artifacts):
    """Return dataframe of employees at risk (>= 30% probability) with Attrition_Risk column."""
    if artifacts is None:
        return None
    try:
        X, _, _, _ = preprocess_data(df)
        proba = artifacts["model"].predict_proba(X)[:, 1]
        at_risk_mask = proba >= 0.3
        result_df = df[at_risk_mask].copy()
        result_df = result_df.reset_index(drop=True)
        result_df.insert(0, "Attrition_Risk_%", (proba[at_risk_mask] * 100).round(1))
        return result_df
    except Exception:
        return None


def get_risk_level(probability: float) -> tuple:
    """Convert attrition probability to risk level and percentage."""
    pct = probability * 100
    if pct < 30:
        return "Low Risk", pct
    elif pct < 60:
        return "Medium Risk", pct
    else:
        return "High Risk", pct


def generate_ai_explanation(form_data: dict, risk_pct: float) -> str:
    """Generate AI explanation for the attrition risk prediction."""
    reasons = []
    if form_data.get("OverTime") == "Yes":
        reasons.append("high overtime")
    if form_data.get("MonthlyIncome", 0) < 5000:
        reasons.append("low MonthlyIncome")
    if form_data.get("JobSatisfaction", 3) <= 2:
        reasons.append("low JobSatisfaction")
    if form_data.get("YearsAtCompany", 5) < 3:
        reasons.append("few YearsAtCompany")
    if form_data.get("WorkLifeBalance", 3) <= 2:
        reasons.append("poor WorkLifeBalance")
    if form_data.get("Age", 36) < 35 and risk_pct > 40:
        reasons.append("early-career stage")
    
    if reasons:
        return f"High attrition risk due to {', '.join(reasons)}."
    elif risk_pct > 50:
        return "Elevated attrition risk based on combined factors in the employee profile."
    else:
        return "Moderate risk profile. Key retention factors appear favorable."


def generate_hr_recommendations(form_data: dict, risk_pct: float) -> list:
    """Generate HR recommendations based on risk factors."""
    recommendations = []
    if form_data.get("OverTime") == "Yes":
        recommendations.append("Reduce overtime - consider workload redistribution or additional hires")
    if form_data.get("MonthlyIncome", 10000) < 5000:
        recommendations.append("Increase salary - review compensation to align with market rates")
    if form_data.get("JobSatisfaction", 3) <= 2:
        recommendations.append("Offer promotion opportunities - discuss career growth path")
    if form_data.get("YearsAtCompany", 5) < 3:
        recommendations.append("Increase engagement - mentorship and onboarding for newer employees")
    if form_data.get("WorkLifeBalance", 3) <= 2:
        recommendations.append("Improve work-life balance - flexible hours or remote work options")
    if form_data.get("Age", 36) < 35:
        recommendations.append("Career development - provide learning and growth opportunities")
    if not recommendations and risk_pct > 30:
        recommendations.append("Schedule 1:1 conversation to understand employee needs")
        recommendations.append("Review overall job fit and team dynamics")
    if risk_pct < 30:
        recommendations.append("Continue current retention practices - risk is low")
    return recommendations


def apply_dark_theme(fig):
    """Apply dark theme to Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9", size=13),
        title=dict(font=dict(size=16, color="#ffffff")),
        xaxis=dict(gridcolor="#334155", linecolor="#475569", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", linecolor="#475569", zerolinecolor="#334155"),
    )
    return fig


def render_hr_charts(df):
    """Render the 4 Plotly charts in 2x2 grid with chart type selector for each."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by Department
        dept_attr = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").sum()).reset_index()
        dept_attr.columns = ["Department", "Attrition_Count"]
        chart1_type = st.selectbox(
            "Attrition by Department — Chart Type",
            ["Bar", "Pie", "Line"],
            key="chart_dept"
        )
        if chart1_type == "Bar":
            fig1 = px.bar(dept_attr, x="Department", y="Attrition_Count", 
                          title="Attrition by Department",
                          color="Attrition_Count", color_continuous_scale=["#334155", "#e50914"])
        elif chart1_type == "Pie":
            fig1 = px.pie(dept_attr, values="Attrition_Count", names="Department",
                          title="Attrition by Department", color_discrete_sequence=["#e50914", "#3b82f6", "#22c55e"])
        else:
            fig1 = px.line(dept_attr, x="Department", y="Attrition_Count",
                           title="Attrition by Department", markers=True)
            fig1.update_traces(line_color="#e50914")
        fig1.update_layout(showlegend=(chart1_type == "Pie"), margin=dict(l=20, r=20, t=50, b=20), height=360)
        fig1 = apply_dark_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Age Distribution
        age_bins = pd.cut(df["Age"], bins=10).value_counts().sort_index().reset_index()
        age_bins.columns = ["Age_Range", "Count"]
        age_bins["Age_Range"] = age_bins["Age_Range"].astype(str)
        chart3_type = st.selectbox(
            "Age Distribution — Chart Type",
            ["Histogram", "Bar", "Pie", "Line"],
            key="chart_age"
        )
        if chart3_type == "Histogram":
            fig3 = px.histogram(df, x="Age", nbins=25, title="Age Distribution of Employees",
                               color_discrete_sequence=["#e50914"])
        elif chart3_type == "Bar":
            fig3 = px.bar(age_bins, x="Age_Range", y="Count", title="Age Distribution of Employees",
                          color="Count", color_continuous_scale=["#334155", "#e50914"])
            fig3.update_layout(showlegend=False, xaxis_title="Age Range")
        elif chart3_type == "Pie":
            fig3 = px.pie(age_bins, values="Count", names="Age_Range", title="Age Distribution of Employees",
                          color_discrete_sequence=px.colors.sequential.Reds_r)
        else:
            fig3 = px.line(age_bins, x="Age_Range", y="Count", title="Age Distribution of Employees", markers=True)
            fig3.update_traces(line_color="#e50914")
        fig3.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
        fig3 = apply_dark_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Attrition by Job Role
        role_attr = df.groupby("JobRole")["Attrition"].apply(lambda x: (x == "Yes").sum()).reset_index()
        role_attr.columns = ["JobRole", "Attrition_Count"]
        role_attr = role_attr.sort_values("Attrition_Count", ascending=True)
        chart2_type = st.selectbox(
            "Attrition by Job Role — Chart Type",
            ["Horizontal Bar", "Bar", "Pie", "Line"],
            key="chart_role"
        )
        if chart2_type == "Horizontal Bar":
            fig2 = px.bar(role_attr, x="Attrition_Count", y="JobRole", orientation="h",
                          title="Attrition by Job Role",
                          color="Attrition_Count", color_continuous_scale=["#334155", "#3b82f6"])
        elif chart2_type == "Bar":
            fig2 = px.bar(role_attr, x="JobRole", y="Attrition_Count",
                          title="Attrition by Job Role",
                          color="Attrition_Count", color_continuous_scale=["#334155", "#3b82f6"])
        elif chart2_type == "Pie":
            fig2 = px.pie(role_attr, values="Attrition_Count", names="JobRole",
                          title="Attrition by Job Role", color_discrete_sequence=px.colors.sequential.Blues)
        else:
            fig2 = px.line(role_attr, x="JobRole", y="Attrition_Count",
                           title="Attrition by Job Role", markers=True)
            fig2.update_traces(line_color="#3b82f6")
            fig2.update_layout(xaxis_tickangle=-45)
        fig2.update_layout(showlegend=(chart2_type == "Pie"), margin=dict(l=20, r=20, t=50, b=20), height=360)
        fig2 = apply_dark_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Overtime vs Attrition
        ot_attr = pd.crosstab(df["OverTime"], df["Attrition"])
        ot_attr = ot_attr.reset_index()
        ot_attr_melt = ot_attr.melt(id_vars="OverTime", var_name="Attrition", value_name="Count")
        chart4_type = st.selectbox(
            "Overtime vs Attrition — Chart Type",
            ["Grouped Bar", "Stacked Bar", "Pie", "Line"],
            key="chart_overtime"
        )
        if chart4_type == "Grouped Bar":
            fig4 = px.bar(ot_attr_melt, x="OverTime", y="Count", color="Attrition", barmode="group",
                          title="Overtime vs Attrition",
                          color_discrete_map={"No": "#22c55e", "Yes": "#e50914"})
        elif chart4_type == "Stacked Bar":
            fig4 = px.bar(ot_attr_melt, x="OverTime", y="Count", color="Attrition", barmode="stack",
                          title="Overtime vs Attrition",
                          color_discrete_map={"No": "#22c55e", "Yes": "#e50914"})
        elif chart4_type == "Pie":
            ot_attr_melt["Label"] = ot_attr_melt["OverTime"].astype(str) + " / " + ot_attr_melt["Attrition"].astype(str)
            fig4 = px.pie(ot_attr_melt, values="Count", names="Label",
                          title="Overtime vs Attrition", color_discrete_sequence=["#22c55e", "#e50914", "#3b82f6", "#fbbf24"])
        else:
            fig4 = px.line(ot_attr_melt, x="OverTime", y="Count", color="Attrition",
                           title="Overtime vs Attrition", markers=True,
                           color_discrete_map={"No": "#22c55e", "Yes": "#e50914"})
        fig4.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
        fig4 = apply_dark_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)


# Load data
@st.cache_data
def get_cached_data():
    return load_dataset()

df = get_cached_data()
artifacts = load_model_artifacts()

# ============ SIDEBAR - Clean modern navigation (no radio buttons) ============
nav_options = ["Dashboard", "Dataset Preview", "Train Model", "HR Insights", "Individual Employee Risk Analyzer", "Attrition Data", "Employees At Risk"]
nav_labels = {
    "Dashboard": "Company Dashboard",
    "Dataset Preview": "Dataset Preview",
    "Train Model": "Train Model",
    "HR Insights": "HR Insights",
    "Individual Employee Risk Analyzer": "Individual Employee Risk Analyzer",
    "Attrition Data": "Attrition Data (Employees Who Left)",
    "Employees At Risk": "Employees At Risk",
}

# Handle programmatic navigation (from KPI "View Data" buttons, Employees At Risk row click)
if "goto_section" in st.session_state and st.session_state.goto_section in nav_options:
    st.session_state["page"] = st.session_state.goto_section
    del st.session_state["goto_section"]
elif "page" not in st.session_state or st.session_state.page not in nav_options:
    st.session_state["page"] = "Dashboard"

st.sidebar.markdown("**Navigation**")

# Check button clicks for nav
section = st.session_state["page"]
for key in nav_options:
    if st.sidebar.button(nav_labels[key], key=f"nav_{key}"):
        st.session_state["page"] = key
        section = key
        st.rerun()

section = st.session_state["page"]
st.session_state["current_section"] = section

# Dynamic CSS for active nav item (highlight + bottom underline)
active_idx = nav_options.index(section)
st.sidebar.markdown(f"""
<style>
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child({active_idx + 2}) button {{
    color: #ffffff !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #00BFFF !important;
}}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("AI Workforce Guardian v1.0")
st.sidebar.caption("Predict attrition • Retain talent")

# ============ PREMIUM HEADER ============
st.markdown('<p class="main-header">AI Workforce Guardian</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Company-Level Analytics & Individual Employee Risk Prediction</p>', unsafe_allow_html=True)

# ============ SECTION: Dashboard (Company-Level Analysis) ============
if section == "Dashboard":
    st.markdown('<p class="section-title">Company-Level Attrition Analysis</p>', unsafe_allow_html=True)
    st.caption("Analytics across all employees in the dataset")
    
    total_employees = len(df)
    attrition_count = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_count / total_employees) * 100
    avg_income = round(df["MonthlyIncome"].mean(), 0)
    employees_at_risk = get_employees_at_risk(df, artifacts)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_employees:,}</div>
            <div class="kpi-label">Total Employees</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{attrition_rate:.1f}%</div>
            <div class="kpi-label">Attrition Rate</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Data →", key="btn_attrition_kpi"):
            st.session_state.goto_section = "Attrition Data"
            st.rerun()
    
    with kpi3:
        risk_val = str(employees_at_risk) if employees_at_risk is not None else "—"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{risk_val}</div>
            <div class="kpi-label">Employees At Risk</div>
        </div>
        """, unsafe_allow_html=True)
        if employees_at_risk is not None:
            if st.button("View Data →", key="btn_risk_kpi"):
                st.session_state.goto_section = "Employees At Risk"
                st.rerun()
    
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${avg_income:,.0f}</div>
            <div class="kpi-label">Avg Monthly Income</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Analytics Charts</p>', unsafe_allow_html=True)
    render_hr_charts(df)

# ============ SECTION: Dataset Preview ============
elif section == "Dataset Preview":
    st.markdown('<p class="section-title">📊 Dataset Preview</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Dataset Shape", f"{df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(20), use_container_width=True)

# ============ SECTION: Train Model ============
elif section == "Train Model":
    st.markdown('<p class="section-title">🤖 Train Model</p>', unsafe_allow_html=True)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training Random Forest model..."):
            try:
                from model.train_model import train_and_save_model
                data_path = str(project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
                accuracy = train_and_save_model(data_path)
                st.session_state["model_accuracy"] = accuracy
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    artifacts = load_model_artifacts()
    if artifacts:
        st.success("✅ Model is trained and ready for predictions.")
        acc = st.session_state.get("model_accuracy")
        if acc is None:
            X, y, _, _ = preprocess_data(df)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = artifacts["model"].predict(X_test)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.2%}")
    else:
        st.info("👆 Click 'Train Model' to train the Random Forest classifier.")

# ============ SECTION: HR Insights ============
elif section == "HR Insights":
    st.markdown('<p class="section-title">📈 HR Insights</p>', unsafe_allow_html=True)
    
    # 1. Department Filter
    departments = ["All Departments"] + sorted(df["Department"].unique().tolist())
    selected_dept = st.selectbox("Select Department", departments, key="hr_dept_filter")
    
    # Filter data by department
    if selected_dept == "All Departments":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["Department"] == selected_dept].copy()
    
    st.markdown("---")
    
    # 2. Department Summary Section
    st.markdown("### Department Summary")
    tot_emp = len(df_filtered)
    attr_count = (df_filtered["Attrition"] == "Yes").sum()
    attr_rate = (attr_count / tot_emp * 100) if tot_emp > 0 else 0
    avg_income = round(df_filtered["MonthlyIncome"].mean(), 0) if tot_emp > 0 else 0
    avg_years = round(df_filtered["YearsAtCompany"].mean(), 1) if tot_emp > 0 else 0
    
    sum1, sum2, sum3, sum4 = st.columns(4)
    with sum1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{tot_emp:,}</div>
            <div class="kpi-label">Total Employees</div>
        </div>
        """, unsafe_allow_html=True)
    with sum2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{attr_rate:.1f}%</div>
            <div class="kpi-label">Attrition Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with sum3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${avg_income:,.0f}</div>
            <div class="kpi-label">Avg Monthly Income</div>
        </div>
        """, unsafe_allow_html=True)
    with sum4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_years}</div>
            <div class="kpi-label">Avg Years at Company</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Top Risk Employees
    st.markdown("### Top Risk Employees")
    if artifacts is not None and tot_emp > 0:
        try:
            X_full, _, _, _ = preprocess_data(df)
            X_prep = X_full.loc[df_filtered.index]
            proba = artifacts["model"].predict_proba(X_prep)[:, 1]
            risk_df = df_filtered[["EmployeeNumber", "EmployeeName", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany"]].copy()
            risk_df["Predicted_Risk_%"] = (proba * 100).round(1)
            risk_df = risk_df.sort_values("Predicted_Risk_%", ascending=False)
            risk_df = risk_df.rename(columns={"EmployeeNumber": "Employee ID", "EmployeeName": "Employee Name", "Predicted_Risk_%": "Predicted Risk Score"})
            st.dataframe(risk_df.head(20), use_container_width=True)
        except Exception:
            st.info("Unable to compute risk scores for filtered data.")
    else:
        st.info("Train the model first to see top risk employees.")
    
    st.markdown("---")
    
    # 4. Attrition Causes Analysis (Feature Importance)
    st.markdown("### Attrition Causes Analysis")
    if artifacts is not None and tot_emp > 0:
        model = artifacts["model"]
        feature_columns = artifacts["feature_columns"]
        importance = model.feature_importances_
        idx = np.argsort(importance)[-10:][::-1]
        top_features = [feature_columns[i] for i in idx]
        top_importance = importance[idx]
        fig_fi = go.Figure(go.Bar(
            x=top_importance,
            y=top_features,
            orientation="h",
            marker=dict(
                color=top_importance,
                colorscale=[[0, "#334155"], [0.5, "#e50914"], [1.0, "#fbbf24"]],
                line=dict(width=0)
            )
        ))
        fig_fi.update_layout(
            title="Top 10 Factors Causing Attrition",
            xaxis_title="Importance",
            height=400,
            margin=dict(l=140),
            yaxis=dict(autorange="reversed")
        )
        fig_fi = apply_dark_theme(fig_fi)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Train the model first to see attrition causes.")
    
    st.markdown("---")
    
    # 5. Salary vs Attrition & 6. Experience vs Attrition
    ch1, ch2 = st.columns(2)
    
    with ch1:
        st.markdown("### Salary vs Attrition")
        if tot_emp > 0:
            fig_sal = px.box(df_filtered, x="Attrition", y="MonthlyIncome",
                         title="Monthly Income by Attrition Status",
                         color="Attrition", color_discrete_map={"No": "#22c55e", "Yes": "#e50914"})
            fig_sal.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
            fig_sal = apply_dark_theme(fig_sal)
            st.plotly_chart(fig_sal, use_container_width=True)
        else:
            st.info("No data for selected department")
    
    with ch2:
        st.markdown("### Experience vs Attrition")
        if tot_emp > 0:
            exp_attr = df_filtered.groupby("YearsAtCompany")["Attrition"].apply(
                lambda x: (x == "Yes").sum()
            ).reset_index()
            exp_attr.columns = ["YearsAtCompany", "Attrition_Count"]
            fig_exp = px.bar(exp_attr, x="YearsAtCompany", y="Attrition_Count",
                             title="Attrition by Years at Company",
                             color="Attrition_Count", color_continuous_scale=["#334155", "#e50914"])
            fig_exp.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20), height=350)
            fig_exp = apply_dark_theme(fig_exp)
            st.plotly_chart(fig_exp, use_container_width=True)
        else:
            st.info("No data for selected department")
    
    # Original charts (Attrition by Dept, Job Role, Age, Overtime)
    st.markdown("---")
    st.markdown("### Attrition Breakdown")
    if tot_emp > 0:
        render_hr_charts(df_filtered)
    
    st.markdown("---")
    
    # 7. AI Recommendations
    st.markdown("### AI Recommendations")
    recommendations = []
    if attr_rate > 15:
        recommendations.append("High attrition rate detected — conduct exit interviews to understand root causes")
    overtime_leave = df_filtered[(df_filtered["OverTime"] == "Yes") & (df_filtered["Attrition"] == "Yes")]
    if len(overtime_leave) > len(df_filtered[df_filtered["Attrition"] == "Yes"]) * 0.5:
        recommendations.append("Improve work-life balance — employees with overtime show high attrition; consider workload redistribution")
    low_income_leave = df_filtered[(df_filtered["MonthlyIncome"] < df_filtered["MonthlyIncome"].median()) & (df_filtered["Attrition"] == "Yes")]
    if len(low_income_leave) > len(df_filtered[df_filtered["Attrition"] == "Yes"]) * 0.5:
        recommendations.append("Review salary structure — lower-income employees leave more often; align compensation with market rates")
    low_sat = df_filtered[(df_filtered["JobSatisfaction"] <= 2) & (df_filtered["Attrition"] == "Yes")]
    if len(low_sat) > 0 and attr_rate > 10:
        recommendations.append("Improve job satisfaction programs — offer career development and recognition in this department")
    if avg_years < 5 and attr_rate > 10:
        recommendations.append("Focus on early-career retention — new employees are at risk; strengthen onboarding and mentorship")
    if not recommendations:
        recommendations.append("Attrition patterns look favorable — continue monitoring and maintain current retention practices")
    
    st.markdown(f"""
    <div class="ai-recommendations">
        <h4>🤖 AI-Generated Insights</h4>
        <ul>
            {''.join(f'<li>{r}</li>' for r in recommendations)}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============ SECTION: Individual Employee Risk Analyzer ============
elif section == "Individual Employee Risk Analyzer":
    st.markdown('<p class="section-title">👤 Individual Employee Risk Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Search for an employee by name and analyze their attrition risk.")
    
    artifacts = load_model_artifacts()
    if not artifacts:
        st.warning("Please train the model first (Train Model section).")
    else:
        # Check if redirected from Employees At Risk page with employee to analyze
        name_from_redirect = None
        if "analyze_employee" in st.session_state and st.session_state.analyze_employee:
            name_from_redirect = st.session_state.analyze_employee
            del st.session_state["analyze_employee"]
        
        # Get all employee names from dataset
        all_names = df["EmployeeName"].tolist()
        
        st.subheader("Search Employee")
        search_text = st.text_input(
            "Type to search",
            placeholder="e.g. anu, priya, kumar...",
            key="emp_search",
            help="Type part of a name - matching employees will appear in the dropdown below"
        )
        
        # Filter names as user types
        if search_text and search_text.strip():
            filtered_names = [n for n in all_names if search_text.strip().lower() in n.lower()]
        else:
            filtered_names = []
        
        if filtered_names:
            selected_name = st.selectbox(
                "Select employee",
                options=filtered_names[:100],
                key="emp_select",
                format_func=lambda x: x
            )
        else:
            selected_name = None
            if search_text and search_text.strip():
                st.info("No matching employees. Try a different search.")
            else:
                st.caption("Type at least one character above to see matching employees")
        
        predict_btn = st.button("Predict Attrition Risk", key="predict_btn")
        
        employee_to_analyze = name_from_redirect or (selected_name if predict_btn else None)
        if employee_to_analyze:
            model = artifacts["model"]
            # Get employee row from dataset
            emp_row = df[df["EmployeeName"] == employee_to_analyze].iloc[0]
            age = int(emp_row["Age"])
            department = str(emp_row["Department"])
            job_role = str(emp_row["JobRole"])
            monthly_income = int(emp_row["MonthlyIncome"])
            years_at_company = int(emp_row["YearsAtCompany"])
            overtime = str(emp_row["OverTime"])
            job_satisfaction = int(emp_row["JobSatisfaction"])
            work_life_balance = int(emp_row["WorkLifeBalance"])
            
            # Build feature vector using preprocessed data (same as model training)
            X_full, _, _, feature_columns = preprocess_data(df)
            emp_idx = df[df["EmployeeName"] == employee_to_analyze].index[0]
            X_pred = X_full.iloc[[emp_idx]][feature_columns]
            proba = model.predict_proba(X_pred)[0][1]
            risk_level, risk_pct = get_risk_level(proba)
            pred_class = model.predict(X_pred)[0]
            
            form_data = {
                "Age": age,
                "Department": department,
                "JobRole": job_role,
                "MonthlyIncome": monthly_income,
                "OverTime": overtime,
                "JobSatisfaction": job_satisfaction,
                "YearsAtCompany": years_at_company,
                "WorkLifeBalance": work_life_balance,
            }
            
            st.markdown("---")
            display_name = employee_to_analyze
            st.markdown(f'<p class="section-title">Prediction Results for {display_name}</p>', unsafe_allow_html=True)
            
            # Prediction: Likely to Leave or Likely to Stay
            prediction_text = "Likely to Leave" if pred_class == 1 else "Likely to Stay"
            prediction_color = "#e50914" if pred_class == 1 else "#22c55e"
            st.markdown(f"""
            <div class="kpi-card" style="max-width: 400px;">
                <div class="kpi-value" style="font-size: 1.75rem; color: {prediction_color};">{prediction_text}</div>
                <div class="kpi-label">Prediction</div>
            </div>
            """, unsafe_allow_html=True)
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Probability Score", f"{proba:.2%}")
            with r2:
                risk_class = "risk-high" if risk_pct >= 60 else ("risk-medium" if risk_pct >= 30 else "risk-low")
                st.metric("Risk Level", risk_level)
                st.markdown(f'<p class="{risk_class}">Attrition Risk: {risk_pct:.1f}%</p>', unsafe_allow_html=True)
            with r3:
                st.metric("Attrition Risk %", f"{risk_pct:.1f}%")
            
            st.markdown("### AI Explanation")
            explanation = generate_ai_explanation(form_data, risk_pct)
            st.info(explanation)
            
            st.markdown("### HR Recommendations")
            recommendations = generate_hr_recommendations(form_data, risk_pct)
            for rec in recommendations:
                st.markdown(f"- {rec}")

# ============ SECTION: Attrition Data (Employees Who Left) ============
elif section == "Attrition Data":
    st.markdown('<p class="section-title">📋 Attrition Data — Employees Who Left</p>', unsafe_allow_html=True)
    attrition_df = df[df["Attrition"] == "Yes"].copy()
    st.metric("Total", f"{len(attrition_df)} employees")
    st.caption("Employees who have left the company")
    if len(attrition_df) > 0:
        st.dataframe(attrition_df, use_container_width=True)
    else:
        st.info("No attrition data.")
    if st.button("← Back to Dashboard", key="back_attrition"):
        st.session_state.goto_section = "Dashboard"
        st.rerun()

# ============ SECTION: Employees At Risk ============
elif section == "Employees At Risk":
    st.markdown('<p class="section-title">⚠️ Employees At Risk</p>', unsafe_allow_html=True)
    at_risk_df = get_at_risk_employees_df(df, artifacts)
    if at_risk_df is not None:
        st.metric("Total", f"{len(at_risk_df)} employees at risk (≥30% attrition probability)")
        st.caption("Click a row to view that employee's prediction on the Individual Employee Risk Analyzer page.")
        
        event = st.dataframe(
            at_risk_df,
            key="at_risk_table",
            selection_mode="single-row",
            on_select="rerun",
            use_container_width=True
        )
        if event and hasattr(event, "selection") and event.selection and event.selection.rows:
            row_idx = event.selection.rows[0]
            emp_name = at_risk_df.iloc[row_idx]["EmployeeName"]
            st.session_state.goto_section = "Individual Employee Risk Analyzer"
            st.session_state.analyze_employee = emp_name
            st.rerun()
    else:
        st.warning("Train the model first to see employees at risk.")
    if st.button("← Back to Dashboard", key="back_risk"):
        st.session_state.goto_section = "Dashboard"
        st.rerun()
