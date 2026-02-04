"""
Evaluation Agent for AutoML System
Handles model evaluation tasks with LLM assistance
"""

import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

from .base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage


class EvaluationAgent(BaseAgent):
    """Agent responsible for model evaluation tasks with LLM assistance"""

    def __init__(self):
        super().__init__(
            name="Evaluation Agent",
            description="Handles model evaluation tasks with LLM assistance",
        )

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process model evaluation task with LLM assistance

        Args:
            task: Dictionary containing task details
                - models: Dictionary of trained models
                - test_data: Test data dictionary with X_test, y_test
                - evaluation_metrics: List of metrics to compute

        Returns:
            Dictionary containing evaluation results with LLM-generated insights
        """
        print(f"[{self.name}] Starting model evaluation task...")
        try:
            # Extract task details
            models = task.get("models", {})
            test_data = task.get("test_data", {})

            if not models:
                raise ValueError("At least one model is required for evaluation")

            if not test_data:
                raise ValueError("Test data is required for model evaluation")

            X_test = test_data.get("X_test")
            y_test = test_data.get("y_test")

            if X_test is None or y_test is None:
                raise ValueError("Both X_test and y_test are required for evaluation")

            print(
                f"[{self.name}] Test data shape: X_test={getattr(X_test, 'shape', 'Unknown')}, y_test={getattr(y_test, 'shape', 'Unknown')}"
            )

            # Evaluate models
            print(f"[{self.name}] Evaluating {len(models)} models...")
            evaluation_results = {}

            for model_name, model in models.items():
                print(f"[{self.name}] Evaluating {model_name} model...")
                eval_result = self._evaluate_model(model, X_test, y_test)
                evaluation_results[model_name] = eval_result
                print(
                    f"[{self.name}] {model_name} evaluation completed with accuracy: {eval_result.get('accuracy')}"
                )

            # Use LLM to analyze results and generate insights
            print(f"[{self.name}] Generating insights with LLM...")
            llm_insights = self._generate_insights_with_llm(evaluation_results)

            print(f"[{self.name}] All models evaluated successfully!")
            # Return results
            return {
                "status": "success",
                "message": "Model evaluation completed successfully with LLM assistance",
                "data": {
                    "evaluation_results": evaluation_results,
                    "llm_insights": llm_insights,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model evaluation failed: {str(e)}",
                "data": None,
            }

    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, Any]:
        """Evaluate a single model"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()

            return {
                "accuracy": accuracy,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix,
                "status": "success",
                "message": "Model evaluated successfully",
            }
        except Exception as e:
            return {
                "accuracy": None,
                "classification_report": None,
                "confusion_matrix": None,
                "status": "error",
                "message": f"Failed to evaluate model: {str(e)}",
            }

    def _generate_insights_with_llm(self, evaluation_results: Dict[str, Any]) -> str:
        """Use LLM to generate insights from evaluation results"""
        # In a real implementation, we would use the LLM to analyze the results
        # and generate natural language insights. For now, we'll return a placeholder.

        insights = "LLM would analyze the evaluation results and provide detailed insights here."
        insights += "\nThis would include model comparisons, strengths/weaknesses, and recommendations."

        return insights
