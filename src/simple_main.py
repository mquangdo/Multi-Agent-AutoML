"""
Simple main script to run the AutoML pipeline with iris dataset
This version doesn't rely on external LLM APIs
"""

import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_path):
    """Load and preprocess the iris data"""
    print("Loading and preprocessing data...")

    # Load data
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")

    # Separate features and target
    target_column = df.columns[-1]  # Assuming last column is target
    X = df.iloc[:, :-1]
    y = df[target_column]

    # Encode target labels if they are strings
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train multiple models"""
    print("Training models...")

    models = {}
    training_results = {}

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))

    models["Random Forest"] = rf_model
    training_results["Random Forest"] = {
        "training_accuracy": rf_train_acc,
        "status": "success",
    }

    # Train SVM
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))

    models["SVM"] = svm_model
    training_results["SVM"] = {"training_accuracy": svm_train_acc, "status": "success"}

    print("Model training completed.")
    return models, training_results


def evaluate_models(models, X_test, y_test):
    """Evaluate trained models"""
    print("Evaluating models...")

    evaluation_results = {}

    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        evaluation_results[model_name] = {"accuracy": accuracy, "status": "success"}

    print("Model evaluation completed.")
    return evaluation_results


def generate_report(data_shapes, training_results, evaluation_results):
    """Generate a simple report"""
    print("\n" + "=" * 50)
    print("AutoML Pipeline Completed Successfully!")
    print("=" * 50)

    # Data processing results
    print(f"Data Processing:")
    print(f"  - Original data shape: {data_shapes[0]}")
    print(f"  - Training data shape: {data_shapes[1]}")
    print(f"  - Test data shape: {data_shapes[2]}")
    print()

    # Training results
    print("Model Training Results:")
    for model_name, results in training_results.items():
        accuracy = results.get("training_accuracy", "N/A")
        print(f"  - {model_name}: Training Accuracy = {accuracy:.4f}")
    print()

    # Evaluation results
    print("Model Evaluation Results:")
    for model_name, results in evaluation_results.items():
        accuracy = results.get("accuracy", "N/A")
        print(f"  - {model_name}: Test Accuracy = {accuracy:.4f}")
    print()

    # Conclusion
    print("Pipeline completed successfully!")


def run_automl_pipeline(file_path):
    """Run the complete AutoML pipeline"""
    print(f"Running AutoML pipeline for file: {file_path}")

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    data_shapes = [
        (len(X_train) + len(X_test), len(X_train.columns)),
        (len(X_train), len(X_train.columns)),
        (len(X_test), len(X_test.columns)),
    ]

    # Train models
    models, training_results = train_models(X_train, y_train)

    # Evaluate models
    evaluation_results = evaluate_models(models, X_test, y_test)

    # Generate report
    generate_report(data_shapes, training_results, evaluation_results)

    return {
        "data_shapes": data_shapes,
        "training_results": training_results,
        "evaluation_results": evaluation_results,
    }


if __name__ == "__main__":
    # Check if data file exists
    file_path = "data/iris_data.csv"

    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure you have the iris_data.csv file in the data directory.")
        sys.exit(1)

    # Run the AutoML pipeline
    try:
        result = run_automl_pipeline(file_path)
        print("\nAutoML pipeline completed successfully!")
    except Exception as e:
        print(f"Error running AutoML pipeline: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
