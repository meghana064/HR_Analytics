# AI Workforce Guardian – Employee Attrition Predictor

An AI-powered HR analytics system that predicts employee attrition before it happens. The system identifies employees at risk of leaving, explains the reasons, and provides actionable recommendations for HR teams to improve employee retention.

## Features

- **Attrition Risk Prediction** – Machine learning model to predict which employees are at risk
- **Risk Percentage Indicator** – Clear Low (0–30%), Medium (30–60%), High (60–100%) risk levels
- **Feature Importance Analysis** – Top 10 features driving attrition predictions
- **HR Insights Dashboard** – Visualizations for attrition patterns
- **AI Explanations** – Human-readable reasons for high-risk predictions
- **HR Recommendations** – Actionable suggestions (salary, overtime, promotions, work-life balance)
- **Interactive Employee Risk Analyzer** – Form to predict risk for individual employees

## Project Structure

```
employee_attrition_ai/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── model/
│   ├── train_model.py
│   └── attrition_model.joblib  (created after training)
├── utils/
│   └── preprocessing.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is in `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## Usage

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Workflow

1. **Dataset Preview** – View the IBM HR Analytics dataset
2. **Train Model** – Click "Train Model" to train the Random Forest classifier
3. **HR Insights** – Explore attrition distribution and key factors
4. **Feature Importance** – See which factors most influence attrition
5. **Employee Risk Analyzer** – Enter employee details and predict attrition risk

### Employee Risk Analyzer Form

Enter the following employee details:

- **Age**
- **Monthly Income**
- **Over Time** (Yes/No)
- **Job Satisfaction** (1–4)
- **Years at Company**
- **Work-Life Balance** (1–4)

Click **Predict Attrition Risk** to get:

- Risk percentage and level
- AI explanation of contributing factors
- HR recommendations for retention

## Technologies

- Python
- pandas, numpy
- scikit-learn (RandomForestClassifier)
- matplotlib, seaborn
- Streamlit

## License

This project uses the IBM HR Analytics Employee Attrition dataset for educational and analytics purposes.
