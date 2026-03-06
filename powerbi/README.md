# Power BI — HR Analytics Attrition Dashboard

This folder contains everything needed to build the **AI Workforce Guardian** dashboard in Power BI.

## Files

| File | Purpose |
|------|---------|
| **DASHBOARD_BLUEPRINT.md** | Click-by-click guide — start here (~20 min) |
| **POWER_BI_SETUP_GUIDE.md** | Full setup, DAX, visuals reference |
| **DAX_Measures.txt** | DAX formulas to copy into Power BI |
| **theme_workforce_guardian.json** | Dark theme (View → Themes → Browse) |
| **PowerQuery_EmployeeName.pq** | Power Query snippet for Employee Name |
| **export_predictions_for_powerbi.py** | Export ML predictions (optional) |

## Quick Start

1. Open **Power BI Desktop**.
2. Follow **DASHBOARD_BLUEPRINT.md** step-by-step.
3. Or: Get data → Text/CSV → load `employee_attrition_ai/data/WA_Fn-UseC_-HR-Employee-Attrition.csv` → create measures from **DAX_Measures.txt** → build visuals.

## Optional: Add ML Predictions

To show "Employees At Risk" and "Predicted Risk %" in Power BI:

```bash
# From project root, after training the model
python powerbi/export_predictions_for_powerbi.py
```

Then import `powerbi/at_risk_predictions.csv` in Power BI and relate to main table on `EmployeeNumber`.
