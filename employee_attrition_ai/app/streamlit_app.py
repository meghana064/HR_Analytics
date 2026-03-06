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


def _safe_val(v, default):
    """Handle numpy/nan values."""
    if v is None or (hasattr(v, "__float__") and np.isnan(v)):
        return default
    return v


def generate_ai_explanation(emp_row: dict, df: pd.DataFrame, risk_pct: float) -> str:
    """
    Generate personalized AI explanation for each employee based on their actual profile.
    Uses department/company averages for context; varies phrasing per employee.
    """
    # Compute department and company benchmarks
    dept = str(emp_row.get("Department", ""))
    dept_mask = df["Department"] == dept if dept else pd.Series([False] * len(df))
    dept_df = df[dept_mask] if dept_mask.any() else df
    
    avg_income = float(df["MonthlyIncome"].median())
    dept_income = float(dept_df["MonthlyIncome"].median()) if len(dept_df) > 0 else avg_income
    
    income = int(_safe_val(emp_row.get("MonthlyIncome"), 0))
    job_sat = int(_safe_val(emp_row.get("JobSatisfaction"), 3))
    wlb = int(_safe_val(emp_row.get("WorkLifeBalance"), 3))
    years = int(_safe_val(emp_row.get("YearsAtCompany"), 5))
    years_since_promo = int(_safe_val(emp_row.get("YearsSinceLastPromotion"), 2))
    overtime = str(emp_row.get("OverTime", "No"))
    age = int(_safe_val(emp_row.get("Age"), 36))
    env_sat = int(_safe_val(emp_row.get("EnvironmentSatisfaction"), 3))
    rel_sat = int(_safe_val(emp_row.get("RelationshipSatisfaction"), 3))
    job_inv = int(_safe_val(emp_row.get("JobInvolvement"), 3))
    dist_home = int(_safe_val(emp_row.get("DistanceFromHome"), 10))
    num_companies = int(_safe_val(emp_row.get("NumCompaniesWorked"), 2))
    training = int(_safe_val(emp_row.get("TrainingTimesLastYear"), 2))
    
    reasons = []
    
    # Overtime - vary phrasing
    if overtime == "Yes":
        if income < dept_income * 0.8:
            reasons.append("works overtime while earning below department median")
        else:
            reasons.append("frequent overtime may indicate burnout or overload")
    
    # Income - contextual
    if income < dept_income * 0.7:
        reasons.append(f"salary (${income:,}) is well below department median (${dept_income:,.0f})")
    elif income < avg_income:
        reasons.append("compensation slightly below company average")
    
    # Job satisfaction - nuanced
    if job_sat <= 1:
        reasons.append("very low job satisfaction (1)")
    elif job_sat == 2:
        reasons.append("below-average job satisfaction (2)")
    
    # Work-life balance
    if wlb <= 1:
        reasons.append("poor work-life balance")
    elif wlb == 2 and overtime == "Yes":
        reasons.append("work-life balance strained by overtime")
    
    # Tenure
    if years < 2:
        reasons.append("less than 2 years tenure—early departure risk")
    elif years < 4 and risk_pct > 40:
        reasons.append("mid-tenure (3–4 years) with elevated risk indicators")
    
    # Promotion stagnation
    if years_since_promo >= 5 and risk_pct > 35:
        reasons.append("no promotion in 5+ years—career stagnation concern")
    
    # Age/career stage
    if age < 30 and num_companies > 3:
        reasons.append("early-career with high job-hopping history")
    elif age >= 45 and years < 3:
        reasons.append("experienced hire with short tenure—integration risk")
    
    # Environment & relationships
    if env_sat <= 2 and job_sat <= 2:
        reasons.append("low environment and job satisfaction")
    elif rel_sat <= 1:
        reasons.append("low relationship satisfaction with colleagues/manager")
    
    # Distance and engagement
    if dist_home > 15:
        reasons.append("long commute may affect retention")
    if job_inv <= 2:
        reasons.append("low job involvement")
    if training == 0 and years >= 2:
        reasons.append("no training in past year—development concern")
    
    # Build personalized message
    if risk_pct >= 60:
        if reasons:
            return f"This employee shows high attrition risk. Key concerns: {'; '.join(reasons)}."
        return "High attrition risk from multiple cumulative factors in the profile."
    elif risk_pct >= 30:
        if reasons:
            return f"Moderate risk profile. Factors to watch: {'; '.join(reasons)}."
        return "Moderate risk—no single strong driver, but combined factors warrant attention."
    else:
        positives = []
        if job_sat >= 4:
            positives.append("strong job satisfaction")
        if wlb >= 3:
            positives.append("good work-life balance")
        if years >= 5:
            positives.append("solid tenure")
        if income >= dept_income:
            positives.append("competitive compensation")
        if overtime == "No":
            positives.append("no overtime burden")
        if positives:
            return f"Low risk profile. Retention strengths: {', '.join(positives)}."
        return "Low attrition risk. Profile indicates stable retention factors."


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


def _report_charts(emp_row, df: pd.DataFrame, risk_pct: float, display_name: str):
    """Generate charts for the HR report: Employee vs benchmarks, Risk gauge, Satisfaction radar."""
    dept = emp_row.get("Department", "")
    dept_df = df[df["Department"] == dept] if dept else df
    
    # Benchmark values
    emp_income = int(emp_row.get("MonthlyIncome", 0))
    emp_jsat = int(emp_row.get("JobSatisfaction", 3))
    emp_esat = int(emp_row.get("EnvironmentSatisfaction", 3))
    emp_rsat = int(emp_row.get("RelationshipSatisfaction", 3))
    emp_wlb = int(emp_row.get("WorkLifeBalance", 3))
    emp_jinv = int(emp_row.get("JobInvolvement", 3))
    
    dept_income = dept_df["MonthlyIncome"].median() if len(dept_df) > 0 else df["MonthlyIncome"].median()
    comp_income = df["MonthlyIncome"].median()
    
    # Chart 1: Employee vs Department/Company - Satisfaction metrics (1-4 scale) + Income (normalized to 0-4)
    metrics = ["Job Satisfaction", "Environment Sat.", "Relationship Sat.", "Work-Life Balance", "Job Involvement"]
    emp_vals = [emp_jsat, emp_esat, emp_rsat, emp_wlb, emp_jinv]
    dept_vals = [
        dept_df["JobSatisfaction"].mean() if len(dept_df) > 0 else df["JobSatisfaction"].mean(),
        dept_df["EnvironmentSatisfaction"].mean() if len(dept_df) > 0 else df["EnvironmentSatisfaction"].mean(),
        dept_df["RelationshipSatisfaction"].mean() if len(dept_df) > 0 else df["RelationshipSatisfaction"].mean(),
        dept_df["WorkLifeBalance"].mean() if len(dept_df) > 0 else df["WorkLifeBalance"].mean(),
        dept_df["JobInvolvement"].mean() if len(dept_df) > 0 else df["JobInvolvement"].mean(),
    ]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="Employee", x=metrics, y=emp_vals, marker_color="#e50914"))
    fig1.add_trace(go.Bar(name="Dept/Company Avg", x=metrics, y=[round(v, 1) for v in dept_vals], marker_color="#3b82f6"))
    fig1.update_layout(
        title=f"{display_name} — Satisfaction vs Department Average",
        barmode="group",
        height=320,
        margin=dict(l=20, r=20, t=50, b=80),
        yaxis=dict(range=[0, 4.5], title="Score (1-4)"),
    )
    fig1 = apply_dark_theme(fig1)
    
    # Chart 2: Income comparison
    fig2 = go.Figure(go.Bar(
        x=["Employee", "Dept Median", "Company Median"],
        y=[emp_income, dept_income, comp_income],
        marker_color=["#e50914", "#3b82f6", "#22c55e"],
        text=[f"${emp_income:,}", f"${dept_income:,.0f}", f"${comp_income:,.0f}"],
        textposition="outside",
    ))
    fig2.update_layout(
        title="Monthly Income Comparison",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    fig2 = apply_dark_theme(fig2)
    
    # Chart 3: Risk gauge
    risk_color = "#22c55e" if risk_pct < 30 else ("#eab308" if risk_pct < 60 else "#e50914")
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 30], "color": "rgba(34, 197, 94, 0.3)"},
                {"range": [30, 60], "color": "rgba(234, 179, 8, 0.3)"},
                {"range": [60, 100], "color": "rgba(229, 9, 20, 0.3)"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 4}, "value": risk_pct},
        },
        title={"text": "Attrition Risk %"},
    ))
    fig3.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    fig3 = apply_dark_theme(fig3)
    
    return fig1, fig2, fig3


def _build_report_html(emp_row, df: pd.DataFrame, risk_pct: float, risk_level: str,
                       explanation: str, recommendations: list, display_name: str) -> str:
    """Build full HTML report for download."""
    fig1, fig2, fig3 = _report_charts(emp_row, df, risk_pct, display_name)
    chart1_html = fig1.to_html(full_html=False, include_plotlyjs="cdn")
    chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)
    
    sections = [
        ("Personal & Demographics", ["EmployeeName", "EmployeeNumber", "Age", "Gender", "MaritalStatus", "DistanceFromHome"]),
        ("Job & Role", ["Department", "JobRole", "JobLevel", "BusinessTravel", "Education", "EducationField"]),
        ("Compensation", ["MonthlyIncome", "MonthlyRate", "DailyRate", "HourlyRate", "PercentSalaryHike", "StockOptionLevel"]),
        ("Satisfaction & Engagement", ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement", "PerformanceRating"]),
        ("Tenure & Experience", ["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked", "TrainingTimesLastYear"]),
        ("Work Pattern & Status", ["OverTime", "Attrition"]),
    ]
    
    rows = []
    for sec_title, cols in sections:
        valid_cols = [c for c in cols if c in emp_row.index]
        if valid_cols:
            rows.append(f'<tr><td colspan="2"><strong>{sec_title}</strong></td></tr>')
            for col in valid_cols:
                val = emp_row[col]
                label = col.replace("_", " ").title()
                if col in ["MonthlyIncome", "MonthlyRate", "DailyRate", "HourlyRate"]:
                    val = f"${val:,.0f}" if isinstance(val, (int, float)) else val
                rows.append(f'<tr><td>{label}</td><td>{val}</td></tr>')
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>HR Report - {display_name}</title>
<style>
body {{ font-family: Segoe UI, sans-serif; background: #0f172a; color: #f1f5f9; padding: 2rem; }}
h1 {{ color: #e50914; }}
h2 {{ color: #94a3b8; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
td {{ padding: 0.5rem; border-bottom: 1px solid #334155; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
.risk {{ color: #e50914; font-weight: bold; }}
.explanation {{ background: #1e293b; padding: 1rem; border-left: 4px solid #e50914; margin: 1rem 0; }}
ul {{ margin-left: 1.5rem; }}
</style>
</head>
<body>
<h1>AI Workforce Guardian — Employee HR Report</h1>
<h2>{display_name}</h2>
<p><strong>Attrition Risk:</strong> <span class="risk">{risk_pct:.1f}%</span> ({risk_level})</p>

<div class="card"><h3>AI Explanation</h3><p>{explanation}</p></div>
<div class="card"><h3>HR Recommendations</h3><ul>{''.join(f'<li>{r}</li>' for r in recommendations)}</ul></div>

<h3>Charts</h3>
<div style="margin: 1rem 0;">{chart1_html}</div>
<div style="margin: 1rem 0;">{chart2_html}</div>
<div style="margin: 1rem 0;">{chart3_html}</div>

<h3>Full Employee Details</h3>
<table><tbody>{''.join(rows)}</tbody></table>
<p style="color: #94a3b8; font-size: 0.9rem;">Generated by AI Workforce Guardian</p>
</body>
</html>"""
    return html


def _render_report_content(emp_row, df, risk_pct, risk_level, explanation, recommendations, display_name):
    """Render report content: charts, details, and download button."""
    st.markdown(f"### {display_name} — Full HR Report")
    risk_class = "risk-high" if risk_pct >= 60 else ("risk-medium" if risk_pct >= 30 else "risk-low")
    st.markdown(f'**Attrition Risk:** <span class="{risk_class}">{risk_pct:.1f}%</span> ({risk_level})', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### AI Explanation")
    st.info(explanation)
    st.markdown("#### HR Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    st.markdown("---")
    st.markdown("#### Charts")
    fig1, fig2, fig3 = _report_charts(emp_row, df, risk_pct, display_name)
    st.plotly_chart(fig1, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Employee Details")
    sections = [
        ("Personal & Demographics", ["EmployeeName", "EmployeeNumber", "Age", "Gender", "MaritalStatus", "DistanceFromHome"]),
        ("Job & Role", ["Department", "JobRole", "JobLevel", "BusinessTravel", "Education", "EducationField"]),
        ("Compensation", ["MonthlyIncome", "MonthlyRate", "DailyRate", "HourlyRate", "PercentSalaryHike", "StockOptionLevel"]),
        ("Satisfaction & Engagement", ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement", "PerformanceRating"]),
        ("Tenure & Experience", ["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked", "TrainingTimesLastYear"]),
        ("Work Pattern & Status", ["OverTime", "Attrition"]),
    ]
    for sec_title, cols in sections:
        valid_cols = [c for c in cols if c in emp_row.index]
        if valid_cols:
            st.markdown(f"**{sec_title}**")
            for col in valid_cols:
                val = emp_row[col]
                label = col.replace("_", " ").title()
                if col in ["MonthlyIncome", "MonthlyRate", "DailyRate", "HourlyRate"] and isinstance(val, (int, float)):
                    st.markdown(f"- **{label}:** ${val:,.0f}")
                else:
                    st.markdown(f"- **{label}:** {val}")
            st.markdown("")
    
    st.markdown("---")
    report_html = _build_report_html(emp_row, df, risk_pct, risk_level, explanation, recommendations, display_name)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in display_name)
    st.download_button(
        "📥 Download Report (HTML)",
        data=report_html,
        file_name=f"HR_Report_{safe_name}.html",
        mime="text/html",
        key=f"btn_download_{safe_name}",
    )


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
nav_options = ["Dashboard", "Dataset Preview", "Train Model", "HR Insights", "Individual Employee Risk Analyzer", "Report", "Attrition Data", "Employees At Risk"]
nav_labels = {
    "Dashboard": "Company Dashboard",
    "Dataset Preview": "Dataset Preview",
    "Train Model": "Train Model",
    "HR Insights": "HR Insights",
    "Individual Employee Risk Analyzer": "Individual Employee Risk Analyzer",
    "Report": "Report",
    "Attrition Data": "Attrition Data (Employees Who Left)",
    "Employees At Risk": "Employees At Risk",
}

# Handle programmatic navigation (from KPI "View Data" buttons, Employees At Risk row click, Report button)
valid_sections = nav_options
if "goto_section" in st.session_state and st.session_state.goto_section in valid_sections:
    st.session_state["page"] = st.session_state.goto_section
    del st.session_state["goto_section"]
elif "page" not in st.session_state or st.session_state.page not in valid_sections:
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
active_idx = nav_options.index(section) if section in nav_options else 0
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
        with st.spinner("Training SVM and Random Forest models..."):
            try:
                from model.train_model import train_and_save_model
                data_path = str(project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
                accuracies = train_and_save_model(data_path)
                st.session_state["model_accuracies"] = accuracies
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    artifacts = load_model_artifacts()
    if artifacts:
        st.success("✅ Models are trained and ready for predictions.")
        accuracies = st.session_state.get("model_accuracies")
        if accuracies is None:
            X, y, _, _ = preprocess_data(df)
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            rf_pred = artifacts["model"].predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            svm_model = SVC(kernel="rbf", C=1.0, random_state=42)
            svm_model.fit(X_train_s, y_train)
            svm_pred = svm_model.predict(X_test_s)
            svm_acc = accuracy_score(y_test, svm_pred)
            accuracies = {"rf": rf_acc, "svm": svm_acc}
        rf_acc = accuracies["rf"]
        svm_acc = accuracies["svm"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest Accuracy", f"{rf_acc:.2%}")
        with col2:
            st.metric("SVM Accuracy", f"{svm_acc:.2%}")
    else:
        st.info("👆 Click 'Train Model' to train SVM and Random Forest classifiers.")

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
            st.session_state["analyzed_employee"] = name_from_redirect
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
        
        # Persist analyzed employee so it survives reruns (e.g. when clicking Report)
        employee_to_analyze = name_from_redirect or (selected_name if predict_btn else None)
        if employee_to_analyze:
            st.session_state["analyzed_employee"] = employee_to_analyze
        employee_to_analyze = st.session_state.get("analyzed_employee") or employee_to_analyze
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
            
            # Build full emp_row dict for AI explanation (personalized per employee)
            emp_row_dict = emp_row.to_dict()
            
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
            explanation = generate_ai_explanation(emp_row_dict, df, risk_pct)
            st.info(explanation)
            
            st.markdown("### HR Recommendations")
            recommendations = generate_hr_recommendations(emp_row_dict, risk_pct)
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Report button - navigates to Report page in nav
            st.markdown("---")
            if st.button("Report", key="btn_full_report", type="primary"):
                st.session_state["page"] = "Report"
                st.session_state["report_employee"] = employee_to_analyze
                st.rerun()

# ============ SECTION: Report (in nav bar - details, charts, download) ============
elif section == "Report":
    st.markdown('<p class="section-title">📋 Employee HR Report</p>', unsafe_allow_html=True)
    
    if artifacts is None:
        st.warning("Please train the model first (Train Model section).")
    else:
        report_employee = st.session_state.get("report_employee")
        
        # If report_employee is set (from Individual Risk Analyzer Report button): show report directly, no name prompt
        if report_employee and report_employee in df["EmployeeName"].values:
            st.caption(f"Report for **{report_employee}**")
            emp_row = df[df["EmployeeName"] == report_employee].iloc[0]
            display_name = report_employee
            X_full, _, _, feature_columns = preprocess_data(df)
            emp_idx = df[df["EmployeeName"] == report_employee].index[0]
            proba = artifacts["model"].predict_proba(X_full.iloc[[emp_idx]][feature_columns])[0][1]
            risk_level, risk_pct = get_risk_level(proba)
            explanation = generate_ai_explanation(emp_row.to_dict(), df, risk_pct)
            recommendations = generate_hr_recommendations(emp_row.to_dict(), risk_pct)
            _render_report_content(emp_row, df, risk_pct, risk_level, explanation, recommendations, display_name)
            if st.button("Select different employee", key="btn_change_report_emp"):
                del st.session_state["report_employee"]
                st.rerun()
        else:
            # No employee selected - show search/select (when user navigates to Report from sidebar directly)
            st.caption("Select an employee to view their full report with charts and download option.")
            all_names = df["EmployeeName"].tolist()
            search_report = st.text_input(
                "Type to search employee",
                placeholder="e.g. anu, priya, kumar...",
                key="report_search",
                help="Search by name to select an employee for the report"
            )
            if search_report and search_report.strip():
                filtered = [n for n in all_names if search_report.strip().lower() in n.lower()]
            else:
                filtered = all_names[:100]
            options_report = filtered[:100] if filtered else all_names[:50]
            selected_for_report = st.selectbox(
                "Select employee",
                options=options_report,
                key="report_emp_select",
                format_func=lambda x: x,
            )
            if st.button("View Report", key="btn_view_report", type="primary"):
                if selected_for_report and options_report:
                    st.session_state["report_employee"] = selected_for_report
                    st.rerun()

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
