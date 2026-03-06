"""
Machine learning model training module for Employee Attrition prediction.
Trains both SVM and RandomForestClassifier, saves the Random Forest model for the dashboard.
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import load_data, preprocess_data


def train_and_save_model(data_path: str = None) -> dict:
    """
    Train both SVM and Random Forest models and save the Random Forest to disk.
    
    Args:
        data_path: Optional path to dataset. Uses default if None.
    
    Returns:
        Dict with keys "rf" and "svm" containing accuracy (float) for each model
    """
    # Load the dataset
    print("Loading dataset...")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X, y, encoders, feature_columns = preprocess_data(df)
    
    # Get default values for form-based predictions (median of each feature)
    defaults = {col: X[col].median() for col in feature_columns}
    
    # Split dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for SVM (SVM is sensitive to feature scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # Train SVM
    print("\nTraining SVM model...")
    svm_model = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=["No Attrition", "Attrition"]))
    print("Classification Report (SVM):")
    print(classification_report(y_test, y_pred_svm, target_names=["No Attrition", "Attrition"]))
    
    # Save Random Forest model and artifacts for the Streamlit app (RF used for predictions)
    model_dir = project_root / "model"
    model_dir.mkdir(exist_ok=True)
    artifacts = {
        "model": rf_model,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "defaults": defaults,
    }
    model_file = model_dir / "attrition_model.joblib"
    joblib.dump(artifacts, model_file)
    print("\nModel saved successfully (Random Forest used for predictions).")
    
    return {"rf": rf_accuracy, "svm": svm_accuracy}


if __name__ == "__main__":
    accuracies = train_and_save_model()
    print(f"\nTraining complete.")
    print(f"Random Forest Accuracy: {accuracies['rf']:.2%}")
    print(f"SVM Accuracy: {accuracies['svm']:.2%}")
