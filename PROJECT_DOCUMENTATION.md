# AI Workforce Guardian — Complete Project Documentation

**Author:** [Your Name]  
**Project:** HR Analytics Employee Attrition Prediction Dashboard  
**Version:** 1.0  
**Purpose:** Comprehensive documentation for hackathon submission — covers every detail from scratch: technologies, code, design decisions, and Q&A so evaluators see the project was built with full understanding.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [Technologies Used](#4-technologies-used)
5. [Project Structure](#5-project-structure)
6. [Dataset Details](#6-dataset-details)
7. [Data Preprocessing & Pipeline](#7-data-preprocessing--pipeline)
8. [Machine Learning Model](#8-machine-learning-model)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Code Walkthrough](#10-code-walkthrough)
11. [Design Decisions](#11-design-decisions)
12. [Hackathon Questions & Answers](#12-hackathon-questions--answers)
13. [Challenges Faced & Solutions](#13-challenges-faced--solutions)
14. [Future Improvements](#14-future-improvements)
15. [Conclusion](#15-conclusion)

---

## 1. Executive Summary

I built **AI Workforce Guardian** — an end-to-end HR analytics platform that predicts employee attrition using machine learning. The system combines a Random Forest classifier with an interactive Streamlit dashboard to help HR teams identify at-risk employees, understand contributing factors, and act on AI-generated recommendations. Every component — from data loading and preprocessing to model training, visualization, and UI — was designed and implemented by me.

---

## 2. Problem Statement

Employee attrition is costly for organizations. Losing talent leads to:

- **Recruitment costs:** Hiring new employees is expensive
- **Knowledge loss:** Experienced employees take institutional knowledge with them
- **Morale impact:** High turnover affects team stability and productivity
- **Business disruption:** Critical roles left vacant can delay projects

HR teams often react *after* employees leave. The goal of this project is to **predict attrition before it happens** so that retention efforts can target the right people at the right time.

---

## 3. Solution Overview

I created a full-stack analytics solution with:

| Component | Purpose |
|----------|---------|
| **Data Pipeline** | Load HR dataset, preprocess, encode categorical variables, handle missing values |
| **ML Model** | Random Forest classifier trained on 35 features to predict attrition (Yes/No) |
| **Dashboard** | 7 sections: Company Dashboard, Dataset Preview, Train Model, HR Insights, Individual Risk Analyzer, Attrition Data, Employees At Risk |
| **Visualizations** | Attrition by department, job role, age, overtime; box plots; feature importance charts |
| **AI Explanations** | Rule-based explanations for high-risk predictions |
| **HR Recommendations** | Actionable suggestions based on employee profile and risk factors |

---

## 4. Technologies Used

### Programming Language
- **Python 3.x** — Core language for data science and web app

### Data Science & ML
- **pandas** (≥1.5.0) — DataFrame operations, data manipulation
- **numpy** (≥1.23.0) — Numerical computations, array operations
- **scikit-learn** (≥1.2.0) — RandomForestClassifier, LabelEncoder, train_test_split, accuracy_score, classification_report
- **joblib** (≥1.2.0) — Save/load trained model artifacts

### Visualization
- **Plotly** (≥5.15.0) — Interactive charts (bar, pie, histogram, box plot)
- **matplotlib** (≥3.6.0) — Fallback/static plotting
- **seaborn** (≥0.12.0) — Statistical visualization support

### Web Framework
- **Streamlit** (≥1.35.0) — Build dashboard without frontend code; widgets, caching, session state

### Development
- **Pathlib** — Cross-platform path handling
- **sys** — Module path management for imports

I chose these technologies because they are industry-standard for data science and allow rapid prototyping. Streamlit lets me focus on logic instead of HTML/CSS/JS.

---

## 5. Project Structure

```
Hr_analytics/
├── employee_attrition_ai/
│   ├── app/
│   │   └── streamlit_app.py      # Main dashboard (871 lines)
│   ├── data/
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── model/
│   │   ├── train_model.py        # Model training script
│   │   ├── __init__.py
│   │   └── attrition_model.joblib # Saved model (created after training)
│   ├── utils/
│   │   ├── preprocessing.py      # Data loading & preprocessing
│   │   └── __init__.py
│   ├── requirements.txt
│   └── README.md
├── requirements.txt              # Root requirements for Streamlit Cloud
├── .gitignore
└── PROJECT_DOCUMENTATION.md      # This file
```

---

## 6. Dataset Details

### Source
**IBM HR Analytics Employee Attrition** dataset (publicly available for analytics).

### File
- **Path:** `employee_attrition_ai/data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Rows:** 1,470 employees
- **Columns:** 35 (before preprocessing)

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| Age | int | Employee age |
| Attrition | Yes/No | Target — did employee leave? |
| BusinessTravel | categorical | Travel frequency |
| Department | categorical | Sales, R&D, HR |
| Education, EducationField | int/categorical | Education level and field |
| JobRole, JobLevel | categorical/int | Role and level |
| JobSatisfaction | 1-4 | Self-reported satisfaction |
| MonthlyIncome | int | Salary |
| OverTime | Yes/No | Works overtime |
| WorkLifeBalance | 1-4 | Self-reported balance |
| YearsAtCompany | int | Tenure |
| YearsInCurrentRole | int | Time in current role |
| ... | ... | Additional HR metrics |

### Target Distribution
- **Attrition = Yes:** ~16% (minority class)
- **Attrition = No:** ~84% (majority class)

I handled this imbalance by using `stratify=y` in `train_test_split` so both classes are proportionally represented in train and test sets.

---

## 7. Data Preprocessing & Pipeline

### File: `utils/preprocessing.py`

#### 7.1 Load Data

I wrote `load_data()` to:
1. Read the CSV with `pd.read_csv()`
2. Add an `EmployeeName` column for UI display — the original dataset has no names, so I generated unique names using predefined first/last name lists and `np.random.choice` with a fixed seed (42) for reproducibility
3. Return the DataFrame

**Code logic:**
```python
def _generate_employee_names(n: int, seed: int = 42) -> list:
    np.random.seed(seed)
    names = set()
    while len(names) < n:
        first = np.random.choice(FIRST_NAMES)
        last = np.random.choice(LAST_NAMES)
        names.add(f"{first} {last}")
    return sorted(names)[:n]
```

#### 7.2 Preprocess Data

`preprocess_data()` does:

1. **Target encoding:** Map `Attrition` Yes→1, No→0
2. **Drop non-predictive columns:** EmployeeCount, EmployeeNumber, StandardHours, Over18, EmployeeName (constant or identifiers)
3. **Missing values:** Mode for categorical, median for numeric
4. **Categorical encoding:** LabelEncoder for object-type columns (Department, JobRole, Gender, etc.)
5. **Feature/target split:** X = all features, y = Attrition

Returns: `(X, y, encoders, feature_columns)` — encoders are saved with the model for inference.

---

## 8. Machine Learning Model

### File: `model/train_model.py`

#### Algorithm
**Random Forest Classifier** — ensemble of decision trees; robust to overfitting, handles mixed feature types, provides feature importance.

#### Hyperparameters
- `n_estimators=100` — 100 trees
- `max_depth=15` — limit tree depth to reduce overfitting
- `random_state=42` — reproducibility

#### Training Steps

1. Load data via `load_data()`
2. Preprocess via `preprocess_data()`
3. Split: 80% train, 20% test, `stratify=y` for class balance
4. Fit `RandomForestClassifier`
5. Evaluate with `accuracy_score` and `classification_report`
6. Save artifacts with `joblib.dump()`:
   - `model` — trained classifier
   - `encoders` — LabelEncoders for categorical columns
   - `feature_columns` — list of feature names
   - `defaults` — median values for each feature (for form-based prediction fallbacks)

#### Output
- **File:** `model/attrition_model.joblib`
- **Metric:** Accuracy on test set (typically ~85–87%)

---

## 9. Streamlit Dashboard

### File: `app/streamlit_app.py`

### 9.1 Page Config
```python
st.set_page_config(
    page_title="AI Workforce Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
I chose `layout="wide"` for more space for charts and tables.

### 9.2 Navigation
I implemented a custom sidebar navigation with:
- Text-based links (no radio buttons)
- Hover effect (underline + color change)
- Active page highlighting via dynamic CSS
- `st.session_state["page"]` for page switching
- `goto_section` for programmatic navigation (e.g., from KPI "View Data" buttons)

### 9.3 Sections

| Section | Description |
|---------|-------------|
| **Company Dashboard** | KPIs (Total Employees, Attrition Rate, Employees At Risk, Avg Income), analytics charts |
| **Dataset Preview** | Shape and first 20 rows |
| **Train Model** | Button to train, shows accuracy when done |
| **HR Insights** | Department filter, summary KPIs, top risk employees, feature importance, salary/experience vs attrition, AI recommendations |
| **Individual Employee Risk Analyzer** | Name search → select employee → predict risk, explanation, recommendations |
| **Attrition Data** | Table of employees who left |
| **Employees At Risk** | Table of employees with ≥30% predicted risk; row click → Individual Risk Analyzer |

### 9.4 Styling
I used custom CSS in `st.markdown(..., unsafe_allow_html=True)`:
- Dark theme (#0f172a background, #1e293b cards)
- Netflix-style accent (#e50914)
- KPI cards with hover effects
- Risk levels: green (low), yellow (medium), red (high)

---

## 10. Code Walkthrough

### 10.1 Key Functions in `streamlit_app.py`

| Function | Purpose |
|----------|---------|
| `load_dataset()` | Load CSV via `utils.preprocessing.load_data()` |
| `load_model_artifacts()` | Load saved model from `model/attrition_model.joblib` |
| `get_employees_at_risk(df, artifacts)` | Count employees with ≥30% attrition probability |
| `get_at_risk_employees_df(df, artifacts)` | DataFrame of at-risk employees with risk % |
| `get_risk_level(probability)` | Map probability → (Low/Medium/High, %) |
| `generate_ai_explanation(form_data, risk_pct)` | Rule-based explanation (overtime, salary, satisfaction, etc.) |
| `generate_hr_recommendations(form_data, risk_pct)` | Actionable retention suggestions |
| `apply_dark_theme(fig)` | Apply dark colors to Plotly figures |
| `render_hr_charts(df)` | 4 charts: attrition by dept, by job role, age distribution, overtime vs attrition |

### 10.2 Caching
```python
@st.cache_data
def get_cached_data():
    return load_dataset()
df = get_cached_data()
```
I use `@st.cache_data` so the dataset is loaded once and reused across reruns.

### 10.3 Individual Risk Analyzer Flow
1. User searches by name (e.g., "anu")
2. Filtered names populate a selectbox
3. User selects employee and clicks "Predict Attrition Risk"
4. Fetch employee row from DataFrame
5. Preprocess to get feature vector (same pipeline as training)
6. `model.predict_proba(X)[:, 1]` → attrition probability
7. Display prediction, risk level, AI explanation, HR recommendations

### 10.4 Employees At Risk → Individual Analyzer
I use `st.dataframe(..., selection_mode="single-row", on_select="rerun")`. On row selection:
```python
row_idx = event.selection.rows[0]
emp_name = at_risk_df.iloc[row_idx]["EmployeeName"]
st.session_state.goto_section = "Individual Employee Risk Analyzer"
st.session_state.analyze_employee = emp_name
st.rerun()
```

---

## 11. Design Decisions

| Decision | Rationale |
|----------|------------|
| **Random Forest** | Handles mixed features, gives feature importance, robust to overfitting |
| **LabelEncoder** | Simple, works for unseen categories in production (with fallback) |
| **Stratified split** | Preserves class ratio in train/test |
| **30% threshold for "at risk"** | Business rule: medium+ risk warrants attention |
| **Custom sidebar nav** | Avoid radio buttons; modern, link-style UI |
| **Dark theme** | Easier on eyes; fits analytics dashboard aesthetic |
| **Generated names** | Original data has no names; needed for search UX |
| **Rule-based AI explanation** | Interpretable; no need for SHAP/LIME in v1 |

---

## 12. Hackathon Questions & Answers

### Q1. Why did you choose Random Forest over other algorithms?

**A:** I compared several options:
- **Logistic Regression:** Simple but linear; HR data has complex interactions
- **Decision Tree:** Prone to overfitting
- **XGBoost/LightGBM:** Strong but more tuning; Random Forest worked well out of the box
- **Neural Networks:** Overkill for this dataset size; harder to interpret

Random Forest gives good accuracy, provides feature importance for HR insights, and doesn't require heavy hyperparameter tuning. I used `max_depth=15` to control overfitting.

---

### Q2. How did you handle the class imbalance (16% attrition vs 84% no attrition)?

**A:** I used multiple strategies:

1. **Stratified split:** `train_test_split(..., stratify=y)` so both classes are represented proportionally in train and test.
2. **Ensemble nature of Random Forest:** Each tree sees bootstrap samples; minority class gets adequate representation across trees.
3. **No SMOTE/oversampling:** With ~235 positive samples, the dataset is large enough that stratification was sufficient. I could add SMOTE in a future version if needed.
4. **Metrics:** I rely on accuracy and could add precision/recall for a more complete picture.

---

### Q3. What features contribute most to attrition? How do you know?

**A:** Random Forest provides `feature_importances_` via Gini impurity. I use this in the HR Insights section:

```python
importance = model.feature_importances_
idx = np.argsort(importance)[-10:][::-1]
top_features = [feature_columns[i] for i in idx]
```

Typical top features: OverTime, MonthlyIncome, JobSatisfaction, YearsAtCompany, WorkLifeBalance, Age. These match HR intuition: overtime, pay, satisfaction, tenure, and work-life balance drive turnover.

---

### Q4. How does the AI explanation work?

**A:** I implemented a **rule-based system** in `generate_ai_explanation()`:

- **OverTime == "Yes"** → "high overtime"
- **MonthlyIncome < 5000** → "low MonthlyIncome"
- **JobSatisfaction ≤ 2** → "low JobSatisfaction"
- **YearsAtCompany < 3** → "few YearsAtCompany"
- **WorkLifeBalance ≤ 2** → "poor WorkLifeBalance"
- **Age < 35 and risk > 40%** → "early-career stage"

If any rules match, I combine them: e.g., "High attrition risk due to high overtime, low MonthlyIncome." If none match but risk > 50%, I use a generic message. Otherwise, "Moderate risk profile."

---

### Q5. Why add EmployeeName? The dataset doesn't have names.

**A:** For the Individual Employee Risk Analyzer I needed a way for users to search and select employees. The original data only has EmployeeNumber. I generated unique names from predefined first/last name lists with a fixed seed so the mapping is reproducible across runs.

---

### Q6. How do you ensure the preprocessing is consistent between training and inference?

**A:** The same `preprocess_data()` function is used in:
- `train_model.py` during training
- `streamlit_app.py` when predicting for an employee

The encoders (LabelEncoder objects) are saved in the joblib file with the model. When predicting for a new employee, I use the *existing* DataFrame and `preprocess_data()`, so the employee row goes through the same pipeline. For form-based input (future), I would use the saved encoders to transform new data.

---

### Q7. What is the 30% threshold for "at risk"?

**A:** I defined:
- **Low Risk:** &lt; 30%
- **Medium Risk:** 30–60%
- **High Risk:** ≥ 60%

30% is a business rule: employees with ≥30% predicted probability are flagged for HR attention. This can be tuned based on retention budget and capacity.

---

### Q8. How did you design the UI/UX?

**A:** 
- **Dark theme:** Reduces eye strain for long analytics sessions; common in dashboards
- **KPI cards:** Key metrics at a glance; "View Data" buttons for drill-down
- **Sidebar nav:** Clean text links, hover effects, active state for clarity
- **Department filter in HR Insights:** Allows segment-level analysis
- **Row selection in Employees At Risk:** One click to see detailed prediction for that employee

---

### Q9. Why Streamlit and not Flask/Django?

**A:** Streamlit lets me build a data app quickly with minimal frontend code. It provides:
- Built-in widgets (buttons, selectboxes, dataframes)
- `st.cache_data` for performance
- `st.session_state` for state
- Native integration with pandas and Plotly

For a hackathon/MVP, speed mattered. Flask/Django would require more setup (templates, APIs, etc.).

---

### Q10. How would you deploy this in production?

**A:** 
1. **Streamlit Community Cloud:** Already set up; connect GitHub repo and specify `employee_attrition_ai/app/streamlit_app.py`
2. **Docker:** Containerize with `streamlit run` as entrypoint
3. **Scaling:** Streamlit Cloud handles it; for self-hosted, use multiple workers/load balancer
4. **Data updates:** Re-train periodically; or add an API to trigger retraining
5. **Security:** Add authentication (Streamlit has experimental support); restrict access to HR team

---

### Q11. What would you improve in v2?

**A:**
- **SHAP/LIME** for model interpretability
- **More models** (XGBoost, Logistic Regression) with comparison
- **Hyperparameter tuning** (GridSearchCV)
- **Time-based split** if we had dates
- **Real-time retraining** when new attrition data arrives
- **Export reports** (PDF/Excel) for HR

---

### Q12. How do you handle a new employee not in the dataset?

**A:** Currently the Individual Risk Analyzer works only for employees in the dataset (name search). For new employees, I would add a form to input Age, Department, JobRole, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, WorkLifeBalance, etc., and use the saved encoders to transform and predict. The `defaults` in the model artifacts are for filling any missing form fields.

---

### Q13. What is the accuracy of your model?

**A:** Typically ~85–87% on the test set. Exact value depends on the random seed and data split. I display it in the Train Model section after training.

---

### Q14. Why drop EmployeeCount, StandardHours, Over18?

**A:** These are constant for all rows (e.g., EmployeeCount=1, StandardHours=80, Over18=Y). They have zero variance and no predictive power. Including them would add noise and slow training.

---

### Q15. Explain the project flow end-to-end.

**A:** 
1. User opens dashboard → data loaded from CSV, cached
2. User goes to Train Model → clicks "Train Model" → `train_model.py` runs → model saved
3. Dashboard loads model from disk
4. Company Dashboard shows KPIs and charts
5. HR Insights: filter by department, see top risk employees, feature importance, AI recommendations
6. Individual Risk Analyzer: search name, select, predict → get probability, explanation, recommendations
7. Employees At Risk: view table, click row → redirected to Individual Risk Analyzer for that employee

---

### Q16. What is `project_root` and why do you use it?

**A:** `project_root = Path(__file__).parent.parent` gets the `employee_attrition_ai` directory. I use it for:
- `project_root / "data" / "file.csv"` — dataset path
- `project_root / "model" / "attrition_model.joblib"` — model path
- `sys.path.insert(0, str(project_root))` — so `from utils.preprocessing import ...` works

This keeps paths correct whether I run from project root or `app/` directory.

---

### Q17. Why use `st.session_state`?

**A:** Streamlit reruns the script on every interaction. To remember the current page, button clicks, or redirects (e.g., from KPI "View Data" to Attrition Data), I need persistent state. `st.session_state` stores values across reruns. I use it for:
- `page` — current section
- `goto_section` — programmatic navigation
- `analyze_employee` — employee name when coming from Employees At Risk row click
- `model_accuracy` — to avoid recomputing after training

---

### Q18. How does the department filter work in HR Insights?

**A:** I use `st.selectbox("Select Department", departments)` where departments = ["All Departments"] + unique values. If "All Departments", I use full `df`. Otherwise I filter: `df_filtered = df[df["Department"] == selected_dept]`. All subsequent charts, KPIs, and risk tables use `df_filtered`.

---

### Q19. What are the risk levels and how are they computed?

**A:** I defined three bands:
- **Low:** probability &lt; 0.30 (30%)
- **Medium:** 0.30 ≤ probability &lt; 0.60
- **High:** probability ≥ 0.60

Implemented in `get_risk_level(probability)` which returns a tuple `(level_name, percentage)`.

---

### Q20. How do you generate HR recommendations?

**A:** In `generate_hr_recommendations()` I use if-else rules:
- OverTime → "Reduce overtime — workload redistribution"
- Low MonthlyIncome → "Review compensation"
- Low JobSatisfaction → "Offer promotion opportunities"
- Few YearsAtCompany → "Mentorship and onboarding"
- Poor WorkLifeBalance → "Flexible hours or remote work"
- Young age → "Career development opportunities"
- If no specific rules match but risk &gt; 30% → "Schedule 1:1 to understand needs"
- If risk &lt; 30% → "Continue current retention practices"

---

### Q21. Why `@st.cache_data`?

**A:** Loading 1,470 rows from CSV and preprocessing runs every time the script reruns. `@st.cache_data` caches the return value so `get_cached_data()` runs only once per session. Same for the dataset — no need to reload on every click.

---

### Q22. Explain the chart type selector (Bar, Pie, Line).

**A:** In `render_hr_charts()` I added `st.selectbox` for each chart. User can switch between Bar, Pie, Line (or Histogram for age). I use `if chart_type == "Bar":` etc. to build the right Plotly figure. This makes the dashboard more exploratory.

---

### Q23. What columns are dropped and why?

**A:** `EmployeeCount`, `EmployeeNumber`, `StandardHours`, `Over18`, `EmployeeName`:
- **EmployeeCount:** Always 1
- **EmployeeNumber:** Identifier, not predictive
- **StandardHours:** Always 80
- **Over18:** Always Y (constant)
- **EmployeeName:** Display-only, added by me; not in original features

---

### Q24. How do you apply dark theme to Plotly charts?

**A:** In `apply_dark_theme(fig)`:
```python
fig.update_layout(
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font=dict(color="#f1f5f9"),
    xaxis=dict(gridcolor="#334155", linecolor="#475569"),
    yaxis=dict(gridcolor="#334155", linecolor="#475569"),
)
```

---

### Q25. What if the model file doesn't exist yet?

**A:** `load_model_artifacts()` returns `None` if the file is missing. The UI checks `if artifacts is None` and shows messages like "Train the model first" or disables risk-related features until the user trains.

---

## 13. Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| No employee names in dataset | Generated unique names with predefined lists + fixed seed |
| Class imbalance | Stratified train-test split |
| Sidebar looked unprofessional | Replaced radio with custom button-based nav + CSS |
| Large DataFrame in Streamlit | Used `@st.cache_data` for loading |
| Consistent preprocessing train vs predict | Same `preprocess_data()` and saved encoders |
| Dark theme for Plotly charts | `apply_dark_theme()` to set bgcolor, font color, grid color |

---

## 14. Future Improvements

1. **Interpretability:** Add SHAP/LIME for per-prediction explanations
2. **Model comparison:** Train multiple models, compare metrics
3. **Hyperparameter tuning:** GridSearchCV or Optuna
4. **New employee form:** Predict for employees not in dataset
5. **Export:** Download charts/tables as PDF/Excel
6. **Authentication:** Login for HR team only
7. **Scheduled retraining:** Cron job to retrain when new data arrives

---

## 15. Conclusion

AI Workforce Guardian is a complete HR analytics solution built from scratch. I designed the data pipeline, trained the Random Forest model, built the Streamlit dashboard with custom UI, and implemented AI explanations and recommendations. The project demonstrates end-to-end ML application development: data → model → deployment → user interface. Every design choice — from algorithm selection to UI styling — was made with the goal of helping HR teams proactively retain talent.

---

## Appendix A: Key Code Snippets

### A.1 Data Loading with Name Generation
```python
def load_data(data_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    names = _generate_employee_names(len(df))
    df.insert(0, "EmployeeName", names)
    return df
```

### A.2 Preprocessing Pipeline
```python
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
cols_to_drop = ["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18", "EmployeeName"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
for col in categorical_cols:
    if col != "Attrition":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
y = df["Attrition"]
X = df.drop(columns=["Attrition"])
```

### A.3 Model Training
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
artifacts = {"model": model, "encoders": encoders, "feature_columns": feature_columns}
joblib.dump(artifacts, model_file)
```

### A.4 Prediction for Individual Employee
```python
X_full, _, _, feature_columns = preprocess_data(df)
emp_idx = df[df["EmployeeName"] == employee_to_analyze].index[0]
X_pred = X_full.iloc[[emp_idx]][feature_columns]
proba = model.predict_proba(X_pred)[0][1]
```

### A.5 At-Risk Employees (≥30% probability)
```python
proba = artifacts["model"].predict_proba(X)[:, 1]
at_risk_mask = proba >= 0.3
result_df = df[at_risk_mask].copy()
result_df.insert(0, "Attrition_Risk_%", (proba[at_risk_mask] * 100).round(1))
```

---

## Appendix B: Installation & Run Commands

```bash
# Clone repository
git clone https://github.com/meghana064/HR_Analytics.git
cd HR_Analytics

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run dashboard (from HR_Analytics or employee_attrition_ai)
cd employee_attrition_ai
streamlit run app/streamlit_app.py

# Or from project root
streamlit run employee_attrition_ai/app/streamlit_app.py
```

---

## Appendix C: File-by-File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `streamlit_app.py` | ~871 | Main dashboard, all 7 sections, styling, nav |
| `train_model.py` | ~86 | Load, preprocess, train, save model |
| `preprocessing.py` | ~143 | load_data, preprocess_data, name generation |
| `requirements.txt` | 8 | Dependencies with versions |

---

## Appendix D: Full Dataset Columns (35)

Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.

---

## Appendix E: Quick Reference — Where Is X?

| I want to... | Location |
|--------------|----------|
| Change risk threshold (30%) | `streamlit_app.py` — `get_employees_at_risk`, `get_risk_level` |
| Add new chart | `streamlit_app.py` — `render_hr_charts()` |
| Change model hyperparameters | `train_model.py` — `RandomForestClassifier(n_estimators=..., max_depth=...)` |
| Add/remove features | `preprocessing.py` — `cols_to_drop`, categorical handling |
| Modify AI explanation rules | `streamlit_app.py` — `generate_ai_explanation()` |
| Change UI colors | `streamlit_app.py` — CSS in first `st.markdown()` block |
| Add new nav section | `streamlit_app.py` — `nav_options`, `nav_labels`, add `elif section == "NewSection"` |

---

**End of Documentation**
