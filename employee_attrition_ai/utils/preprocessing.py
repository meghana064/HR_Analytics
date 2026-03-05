"""
Data preprocessing module for Employee Attrition prediction.
Handles data loading, cleaning, encoding, and feature-target separation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


# Predefined first and last names for generating unique employee names
FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Ameya", "Ananya", "Aniket", "Anisha", "Anuj", "Anupama", "Anusha",
    "Arjun", "Aryan", "Ashwin", "Bhargav", "Bhavya", "Chaitanya", "Deepa", "Devika", "Ganesh", "Harsha",
    "Ishita", "Kavya", "Krishna", "Madhav", "Meera", "Neha", "Nikhil", "Pooja", "Priya", "Rahul",
    "Rajesh", "Riya", "Rohan", "Sahil", "Sanjay", "Shreya", "Sneha", "Varun", "Vikram", "Vishal",
    "Abhishek", "Amit", "Anil", "Deepak", "Kiran", "Lakshmi", "Manish", "Prakash", "Suresh", "Sunita",
]
LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Reddy", "Rao", "Nair", "Menon", "Iyer", "Pillai",
    "Gupta", "Mehta", "Joshi", "Desai", "Shah", "Pandey", "Mishra", "Jain", "Agarwal", "Malhotra",
    "Verma", "Saxena", "Chaturvedi", "Dubey", "Tiwari", "Yadav", "Khan", "Rao", "Nair", "Menon",
    "Kulkarni", "Murthy", "Krishnan", "Sundaram", "Venkatesh", "Bhat", "Shetty", "Pai", "Kamath", "Hegde",
    "Gowda", "Naidu", "Chowdhury", "Bose", "Banerjee", "Mukherjee", "Das", "Ghosh", "Roy", "Sen",
]


def _generate_employee_names(n: int, seed: int = 42) -> list:
    """Generate n unique random names for employees (reproducible)."""
    np.random.seed(seed)
    names = set()
    while len(names) < n:
        first = np.random.choice(FIRST_NAMES)
        last = np.random.choice(LAST_NAMES)
        names.add(f"{first} {last}")
    return sorted(names)[:n]


def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the HR Employee Attrition dataset from CSV.
    Assigns random names to each employee for identification.
    
    Args:
        data_path: Path to the CSV file. Defaults to project data folder.
    
    Returns:
        DataFrame with raw employee data including EmployeeName column
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    df = pd.read_csv(data_path)
    # Assign random names (reproducible based on row order)
    names = _generate_employee_names(len(df))
    df.insert(0, "EmployeeName", names)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset for machine learning.
    
    - Converts Attrition from Yes/No to 1/0
    - Encodes categorical columns using LabelEncoder
    - Handles missing values
    - Separates features (X) and target (y)
    
    Args:
        df: Raw DataFrame from load_data()
    
    Returns:
        Tuple of (X, y, encoders_dict, feature_columns)
        encoders_dict contains LabelEncoder objects for each categorical column
    """
    df = df.copy()
    
    # Convert Attrition column from Yes/No to numeric (1 and 0)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    
    # Identify categorical columns (object type or few unique values)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Columns to drop - constant, identifier, or display-only columns with no predictive value
    cols_to_drop = ["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18", "EmployeeName"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    # Update categorical list after dropping
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # Handle missing values - fill with mode for categorical, median for numeric
    for col in df.columns:
        if df[col].isnull().any():
            if col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical columns using LabelEncoder
    encoders = {}
    for col in categorical_cols:
        if col != "Attrition" and col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Separate features (X) and target variable (y)
    if "Attrition" in df.columns:
        y = df["Attrition"]
        X = df.drop(columns=["Attrition"])
    else:
        raise ValueError("Attrition column not found in dataset")
    
    feature_columns = X.columns.tolist()
    
    return X, y, encoders, feature_columns


def get_default_values(df: pd.DataFrame) -> dict:
    """
    Get default/median values for each feature.
    Used when predicting from Employee Risk Analyzer form with partial input.
    
    Args:
        df: Preprocessed DataFrame (before Attrition separation)
    
    Returns:
        Dictionary of column -> default value
    """
    defaults = {}
    for col in df.columns:
        if col == "Attrition":
            continue
        if df[col].dtype in ["int64", "float64"]:
            defaults[col] = df[col].median()
        else:
            defaults[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
    return defaults
