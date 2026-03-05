"""
Machine learning model training module for Employee Attrition prediction.
Trains a RandomForestClassifier and saves the model for use in the dashboard.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import load_data, preprocess_data


def train_and_save_model(data_path: str = None) -> float:
    """
    Train the Random Forest model and save it to disk.
    
    Args:
        data_path: Optional path to dataset. Uses default if None.
    
    Returns:
        Model accuracy on test set (float)
    """
    # Load the dataset
    print("Loading dataset...")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"First few rows:\n{df.head()}")
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X, y, encoders, feature_columns = preprocess_data(df)
    
    # Get default values for form-based predictions (median of each feature)
    defaults = {col: X[col].median() for col in feature_columns}
    
    # Split dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train RandomForestClassifier
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))
    
    # Save model and artifacts for the Streamlit app
    model_dir = project_root / "model"
    model_dir.mkdir(exist_ok=True)
    
    artifacts = {
        "model": model,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "defaults": defaults,
    }
    
    model_file = model_dir / "attrition_model.joblib"
    joblib.dump(artifacts, model_file)
    print("\nModel saved successfully.")
    
    return accuracy


if __name__ == "__main__":
    # Run training when script is executed directly
    accuracy = train_and_save_model()
    print(f"\nTraining complete. Final accuracy: {accuracy:.2%}")
