"""
Export ML predictions for use in Power BI.
Run from project root: python powerbi/export_predictions_for_powerbi.py
"""

import sys
from pathlib import Path

# Run from Hr_analytics folder; powerbi/ is inside Hr_analytics
_project_root = Path(__file__).resolve().parent.parent
project_root = _project_root / "employee_attrition_ai"
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from utils.preprocessing import load_data, preprocess_data

def export_predictions():
    data_path = project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    model_path = project_root / "model" / "attrition_model.joblib"
    
    if not model_path.exists():
        print("Model not found. Train the model first (Streamlit app → Train Model).")
        return
    
    df = load_data(str(data_path))
    X, _, _, feature_columns = preprocess_data(df)
    artifacts = joblib.load(model_path)
    proba = artifacts["model"].predict_proba(X)[:, 1]
    
    out = df[["EmployeeNumber", "EmployeeName", "Department", "JobRole", "Attrition", "MonthlyIncome", "YearsAtCompany"]].copy()
    out["Predicted_Risk_Pct"] = (proba * 100).round(1)
    out["At_Risk_Flag"] = proba >= 0.3
    
    output_path = Path(__file__).parent / "at_risk_predictions.csv"
    out.to_csv(output_path, index=False)
    print(f"Exported {len(out)} rows to {output_path}")
    print(f"Employees at risk (>=30%): {(out['At_Risk_Flag']).sum()}")

if __name__ == "__main__":
    export_predictions()
