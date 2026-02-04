"""
Data utility functions for AutoML Agent System
"""

import pandas as pd
from typing import Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file path"""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and JSON are supported.")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing of data"""
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Handle categorical variables
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes

    return df


def split_data(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2
) -> Dict[str, Any]:
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
