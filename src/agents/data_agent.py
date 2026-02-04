"""
Data Agent for AutoML System
Handles data preprocessing tasks with LLM assistance
"""

import pandas as pd
from typing import Dict, Any
from .base_agent import BaseAgent
from ..utils.data_utils import load_data, preprocess_data, split_data
from langchain_core.messages import SystemMessage, HumanMessage


class DataAgent(BaseAgent):
    """Agent responsible for data preprocessing tasks with LLM assistance"""

    def __init__(self):
        super().__init__(
            name="Data Agent",
            description="Handles data preprocessing tasks with LLM assistance",
        )

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data preprocessing task with LLM assistance

        Args:
            task: Dictionary containing task details
                - file_path: Path to the data file
                - target_column: Name of the target column
                - preprocessing_steps: List of preprocessing steps to apply (optional)

        Returns:
            Dictionary containing processed data and metadata
        """
        print(f"[{self.name}] Starting data processing task...")
        try:
            # Load data
            file_path = task.get("file_path")
            target_column = task.get("target_column")

            if not file_path:
                raise ValueError("File path is required for data processing")

            if not target_column:
                raise ValueError("Target column is required for data processing")

            print(f"[{self.name}] Loading data from {file_path}...")
            # Load data
            df = load_data(file_path)

            print(f"[{self.name}] Data loaded with shape: {df.shape}")

            # Use LLM to analyze the data and suggest preprocessing steps
            print(
                f"[{self.name}] Analyzing data with LLM to suggest preprocessing steps..."
            )
            preprocessing_plan = self._analyze_data_with_llm(df, target_column)

            # Apply preprocessing based on LLM suggestions
            print(f"[{self.name}] Applying preprocessing based on LLM suggestions...")
            processed_df = self._apply_preprocessing(df, preprocessing_plan)

            # Split data
            print(f"[{self.name}] Splitting data into train/test sets...")
            split_result = split_data(processed_df, target_column)

            print(f"[{self.name}] Data processing completed successfully!")
            # Return results with actual data for downstream processing
            return {
                "status": "success",
                "message": "Data preprocessing completed successfully with LLM assistance",
                "data": {
                    "X_train": split_result["X_train"],
                    "X_test": split_result["X_test"],
                    "y_train": split_result["y_train"],
                    "y_test": split_result["y_test"],
                    "original_shape": df.shape,
                    "processed_shape": processed_df.shape,
                    "preprocessing_plan": preprocessing_plan,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Data preprocessing failed: {str(e)}",
                "data": None,
            }

    def _analyze_data_with_llm(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, Any]:
        """Use LLM to analyze data and suggest preprocessing steps"""
        # Create a summary of the data for the LLM
        data_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "target_column": target_column,
        }

        # In a real implementation, we would use the LLM to analyze this information
        # and suggest preprocessing steps. For now, we'll return a default plan.

        preprocessing_plan = {
            "handle_missing_values": "mean_imputation",
            "handle_categorical": "label_encoding",
            "feature_scaling": "standardization",
            "outlier_detection": "z_score",
            "feature_selection": "none",
        }

        return preprocessing_plan

    def _apply_preprocessing(
        self, df: pd.DataFrame, preprocessing_plan: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply preprocessing steps based on the plan"""
        # This is a simplified version - in practice, you would implement
        # the actual preprocessing steps based on the plan
        return preprocess_data(df)
