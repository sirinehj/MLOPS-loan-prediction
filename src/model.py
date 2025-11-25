import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from feature_engine.outliers import OutlierTrimmer
import joblib
import os
import json


def prepare_data(df):
    """Clean and encode dataset before training."""

    df["person_age"] = df["person_age"].astype(int)

    skewed_cols = [
        "person_age", "person_income", "person_emp_exp",
        "loan_amnt", "loan_percent_income",
        "cb_person_cred_hist_length", "credit_score"
    ]

    norm_cols = ["loan_int_rate"]

    scaler = StandardScaler()
    normalizer = MinMaxScaler()

    df[skewed_cols] = scaler.fit_transform(df[skewed_cols])
    df[norm_cols] = normalizer.fit_transform(df[norm_cols])

    df["person_education"].replace(
        {
            "High School": 0,
            "Associate": 1,
            "Bachelor": 2,
            "Master": 3,
            "Doctorate": 4,
        },
        inplace=True,
    )

    df["person_gender"] = df["person_gender"].map({"male": 0, "female": 1})

    df["person_home_ownership"] = df["person_home_ownership"].map(
        {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    )

    df["loan_intent"] = df["loan_intent"].map(
        {
            "PERSONAL": 0,
            "EDUCATION": 1,
            "MEDICAL": 2,
            "VENTURE": 3,
            "HOMEIMPROVEMENT": 4,
            "DEBTCONSOLIDATION": 5,
        }
    )

    df["previous_loan_defaults_on_file"] = df[
        "previous_loan_defaults_on_file"
    ].map({"No": 0, "Yes": 1})

    trimmer = OutlierTrimmer(
        capping_method="iqr",
        tail="right",
        variables=[
            "person_age", "person_gender", "person_education",
            "person_income", "person_emp_exp",
            "person_home_ownership", "loan_amnt",
            "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score",
            "previous_loan_defaults_on_file",
        ],
    )

    df_clean = trimmer.fit_transform(df)
    return df_clean


def train(df_clean, test_size=0.2, random_state=42):
    """
    Train the SVC model on the prepared data.
    
    Args:
        df_clean: Cleaned and preprocessed DataFrame
        test_size: Test set size
        random_state: Random state for reproducibility
    
    Returns:
        model: Trained SVC model
        X_test: Test features
        y_test: Test labels
        X_train: Train features
        y_train: Train labels
    """
    # Configure X and y from the cleaned dataframe
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize SVC model
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=random_state
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, X_train, y_train


def evaluate(model, X_test, y_test):
    """
    Evaluate the trained model and generate performance metrics.
    
    Args:
        model: Trained SVC model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }
    
    return metrics


def save_model(model, path="models/svc_model.pkl"):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained SVC model
        path: Path where to save the model
    
    Returns:
        str: Path where model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, path)
    
    return path


def load_model(path="models/svc_model.pkl"):
    """
    Load a trained model from a file.
    
    Args:
        path: Path to the saved model
    
    Returns:
        model: Loaded SVC model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    model = joblib.load(path)
    
    return model


def save_metrics(metrics, path="models/svc_metrics.json"):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        path: Path where to save the metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save metrics
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(path="models/svc_metrics.json"):
    """
    Load evaluation metrics from a JSON file.
    
    Args:
        path: Path to the metrics file
    
    Returns:
        dict: Loaded metrics
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found at {path}")
    
    with open(path, 'r') as f:
        metrics = json.load(f)
    
    return metrics