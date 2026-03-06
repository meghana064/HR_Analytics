# AI Workforce Guardian — Power BI Setup Guide

Recreate the HR Analytics Attrition Dashboard in Power BI. This guide covers data import, transformations, DAX measures, and report design.

---

## Prerequisites

- **Power BI Desktop** (free) — [Download](https://powerbi.microsoft.com/desktop/)
- **Dataset:** `employee_attrition_ai/data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

---

## Step 1: Import Data

1. Open **Power BI Desktop**.
2. **Home** → **Get data** → **Text/CSV**.
3. Browse to: `Hr_analytics\employee_attrition_ai\data\WA_Fn-UseC_-HR-Employee-Attrition.csv`
4. Click **Load** (or **Transform Data** to edit in Power Query first).

---

## Step 2: Power Query Transformations (Optional)

If you want to add an **Employee Name** column (like the Streamlit app), use Power Query:

1. **Home** → **Transform data** (or Edit Queries).
2. Select your query in the left pane.
3. **Add Column** → **Index Column**.
4. **Add Column** → **Custom Column**:

   **Name:** `EmployeeName`  
   **Formula:**
   ```
   Text.Combine({
     List.Random(1){0} * 1000000,
     " ",
     Number.ToText([Index])
   })
   ```
   
   *Simpler option:* Use `"Employee " & Text.From([EmployeeNumber])` to get "Employee 1", "Employee 2", etc.

5. **Home** → **Close & Apply**.

---

## Step 3: Create DAX Measures

Go to **Modeling** tab → **New Measure**. Create these measures one by one:

### KPI Measures

| Measure Name | DAX Formula |
|--------------|-------------|
| **Total Employees** | `Total Employees = COUNTROWS(HR_Data)` |
| **Attrition Count** | `Attrition Count = CALCULATE(COUNTROWS(HR_Data), HR_Data[Attrition] = "Yes")` |
| **Attrition Rate %** | `Attrition Rate % = DIVIDE([Attrition Count], [Total Employees], 0) * 100` |
| **Avg Monthly Income** | `Avg Monthly Income = AVERAGE(HR_Data[MonthlyIncome])` |
| **Employees Who Left** | `Employees Who Left = CALCULATE(COUNTROWS(HR_Data), HR_Data[Attrition] = "Yes")` |
| **Employees Retained** | `Employees Retained = CALCULATE(COUNTROWS(HR_Data), HR_Data[Attrition] = "No")` |

*Note: Replace `HR_Data` with your actual table name if different.*

---

## Step 4: Report Pages & Visuals

### Page 1: Company Dashboard

| Visual | Fields | Notes |
|--------|--------|-------|
| **Card** | Total Employees | KPI |
| **Card** | Attrition Rate % | Format as percentage; add % |
| **Card** | Avg Monthly Income | Format: $#,##0 |
| **Stacked Bar** | Department (axis), Attrition (legend), Count (value) | Attrition by Department |
| **Bar Chart** | JobRole (axis), Attrition Count (value) | Sort descending |
| **Clustered Column** | OverTime (axis), Attrition (legend), Count (value) | Overtime vs Attrition |
| **Histogram** | Age (axis), Count of EmployeeNumber (value) | Age distribution |

### Page 2: HR Insights

| Visual | Fields |
|--------|--------|
| **Slicer** | Department (dropdown or list) |
| **4 Cards** | Total Employees, Attrition Rate %, Avg Monthly Income, Avg Years at Company |
| **Table** | EmployeeNumber, Department, JobRole, MonthlyIncome, YearsAtCompany, Attrition |
| **Clustered Bar** | MonthlyIncome by Attrition |
| **Bar Chart** | YearsAtCompany by Attrition Count |

### Page 3: Attrition Data (Employees Who Left)

| Visual | Fields |
|--------|--------|
| **Slicer** | Department (optional) |
| **Table** | All columns; filter `Attrition = "Yes"` |

### Page 4: Department Summary

| Visual | Fields |
|--------|--------|
| **Matrix** | Department (rows), Attrition (columns), Count |
| **Donut** | Department, Employee Count |
| **Line Chart** | Department, Attrition Rate % |

---

## Step 5: Additional DAX (Department-Filtered)

If you use a Department slicer, these measures work automatically with filters. For **Avg Years at Company**:

```
Avg Years at Company = AVERAGE(HR_Data[YearsAtCompany])
```

---

## Step 6: Formatting & Theme

1. **View** → **Themes** → Choose **Executive** or **Innovation** (dark theme).
2. Or create custom theme: **View** → **Themes** → **Customize current theme**.
3. Recommended colors:
   - Background: `#0f172a`
   - Cards: `#1e293b`
   - Accent: `#e50914`
   - Text: `#f1f5f9`

---

## Step 7: Filters & Slicers

- Add **Department** slicer to all pages (sync slicers: **View** → **Sync slicers**).
- Add **Attrition** slicer on Attrition Data page.
- Use **Filter pane** for JobRole, OverTime, etc.

---

## Step 8: ML Prediction (Alternative)

Power BI does **not** run Python/R models directly in the report. Options:

1. **Pre-compute in Python:** Run your Streamlit model, export predictions to CSV (`EmployeeNumber, Predicted_Risk_Pct`), then import and merge in Power BI.
2. **Python visual:** Use **Get data** → **Python script** (requires Python installed; limited support).
3. **Power BI + Azure ML:** Deploy model to Azure, call from Power BI (advanced).

**Simple approach:** Export at-risk employees from Streamlit app (or run `train_model.py` + a small script to predict and export), save as `predictions.csv`, then merge with main data in Power Query on `EmployeeNumber`.

---

## Quick Reference: Table & Column Names

After import, your table might be named `WA_Fn-UseC_-HR-Employee-Attrition` or you may rename it to `HR_Data`.

**Key columns:**
- `Attrition` — "Yes" / "No"
- `Department` — Sales, Research & Development, Human Resources
- `JobRole` — Sales Executive, Research Scientist, etc.
- `MonthlyIncome` — numeric
- `OverTime` — "Yes" / "No"
- `YearsAtCompany` — numeric
- `Age` — numeric

---

## Export Predictions for Power BI (Optional)

To add **Predicted Risk %** and **Employees At Risk** in Power BI:

1. Run this Python script (saves `at_risk_predictions.csv`):

```python
import pandas as pd
import joblib
from pathlib import Path

project_root = Path("employee_attrition_ai")
model_path = project_root / "model" / "attrition_model.joblib"
data_path = project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(data_path)
sys.path.insert(0, str(project_root))
from utils.preprocessing import preprocess_data
# Add EmployeeName if needed
df["EmployeeName"] = ["Employee " + str(i) for i in range(1, len(df)+1)]

X, y, _, fc = preprocess_data(df)
artifacts = joblib.load(model_path)
proba = artifacts["model"].predict_proba(X)[:, 1]

out = df[["EmployeeNumber", "EmployeeName", "Department", "JobRole", "Attrition"]].copy()
out["Predicted_Risk_Pct"] = (proba * 100).round(1)
out["At_Risk_Flag"] = proba >= 0.3
out.to_csv("powerbi/at_risk_predictions.csv", index=False)
```

2. In Power BI: **Get data** → **Text/CSV** → load `at_risk_predictions.csv`.
3. Create relationship: `HR_Data[EmployeeNumber]` — `at_risk_predictions[EmployeeNumber]`.
4. Add measures: `At Risk Count = COUNTROWS(FILTER(at_risk_predictions, at_risk_predictions[At_Risk_Flag] = TRUE))`.

---

**End of Guide**
