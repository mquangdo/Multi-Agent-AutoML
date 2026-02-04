"""
Manager Graph for AutoML System
Orchestrates the entire AutoML pipeline by delegating tasks to sub-agents
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Annotated
import operator
from langchain_core.messages import SystemMessage, HumanMessage


# Define state schema
class ManagerState(BaseModel):
    user_input: str = Field(default="")
    task_decomposition: Dict[str, Any] = Field(default={})
    data_processing_result: Dict[str, Any] = Field(default={})
    model_training_result: Dict[str, Any] = Field(default={})
    model_evaluation_result: Dict[str, Any] = Field(default={})
    final_output: str = Field(default="")
    messages: Annotated[list, operator.add] = Field(default=[])


# Import subgraphs
from .data_graph import data_app
from .training_graph import training_app
from .eval_graph import eval_app
from ..config.llm_config import get_llm

# Initialize LLM
llm = get_llm()


# Define nodes
def task_decomposition_node(state: ManagerState) -> Dict[str, Any]:
    """Decompose user task into subtasks for each agent using LLM reasoning"""
    # Use LLM to parse and decompose the task
    system_prompt = """
    You are an AI assistant that specializes in decomposing AutoML tasks. 
    Given a user request, you need to extract the following information:
    1. Problem type (classification, regression, clustering)
    2. File path or data source
    3. Target column name
    4. Models to train
    5. Any specific requirements or constraints
    
    Respond in JSON format with the extracted information.
    """

    human_message = f"User request: {state.user_input}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    # In a real implementation, we would use the LLM to parse the request
    # For now, we'll simulate the LLM response with a simplified approach

    # Parse user input to extract key information
    user_input = state.user_input.lower()

    # Extract file path (assuming it's a URL or file path in the input)
    file_path = None
    if "đường dẫn" in user_input or "path" in user_input or "file" in user_input:
        # Simplified extraction - in practice, you'd want a more robust parser
        # This is just for demonstration
        file_path = "data/iris_data.csv"  # Correct path to iris data

    # Extract problem type
    problem_type = "classification"  # Default
    if "phân loại" in user_input or "classification" in user_input:
        problem_type = "classification"

    # Extract models to train
    models = []
    if "svm" in user_input:
        models.append("svm")
    if "random forest" in user_input or "randomforest" in user_input:
        models.append("random_forest")

    # Default models if none specified
    if not models:
        models = ["svm", "random_forest"]

    task_decomposition = {
        "problem_type": problem_type,
        "file_path": file_path,
        "target_column": "target",  # Default target column name
        "models": models,
    }

    return {"task_decomposition": task_decomposition}


def data_processing_node(state: ManagerState) -> Dict[str, Any]:
    """Process data using DataAgent"""
    task = {
        "file_path": state.task_decomposition.get("file_path"),
        "target_column": state.task_decomposition.get("target_column"),
    }

    # Run data processing graph
    result = data_app.invoke({"task": task})

    return {"data_processing_result": result["result"]}


def model_training_node(state: ManagerState) -> Dict[str, Any]:
    """Train models using TrainingAgent"""
    # Get training data from previous step
    training_data = {}
    if state.data_processing_result.get("status") == "success":
        data = state.data_processing_result.get("data", {})
        training_data = {
            "X_train": data.get("X_train"),
            "y_train": data.get("y_train"),
        }

    task = {
        "models": state.task_decomposition.get("models"),
        "training_data": training_data,
    }

    # Run training graph
    result = training_app.invoke({"task": task})

    return {"model_training_result": result["result"]}


def model_evaluation_node(state: ManagerState) -> Dict[str, Any]:
    """Evaluate models using EvaluationAgent"""
    # Get test data from data processing step
    test_data = {}
    if state.data_processing_result.get("status") == "success":
        data = state.data_processing_result.get("data", {})
        test_data = {
            "X_test": data.get("X_test"),
            "y_test": data.get("y_test"),
        }

    # Get trained models from training step
    models = {}
    if state.model_training_result.get("status") == "success":
        data = state.model_training_result.get("data", {})
        models = data.get("models", {})

    task = {"models": models, "test_data": test_data}

    # Run evaluation graph
    result = eval_app.invoke({"task": task})

    return {"model_evaluation_result": result["result"]}


def final_output_node(state: ManagerState) -> Dict[str, Any]:
    """Generate final output for user using LLM to create a natural language report"""
    # Prepare results for LLM summarization
    results_summary = {
        "data_processing": state.data_processing_result,
        "model_training": state.model_training_result,
        "model_evaluation": state.model_evaluation_result,
    }

    # Use LLM to generate a natural language report
    system_prompt = """
    You are an AI assistant that explains AutoML results in clear, understandable language.
    Given the results of data processing, model training, and model evaluation, create a 
    comprehensive report for the user. Include:
    1. Summary of data processing steps
    2. Model training results with metrics
    3. Model evaluation results with comparisons
    4. Recommendations based on the results
    """

    human_message = f"AutoML Results: {results_summary}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    # In a real implementation, we would use the LLM to generate the report
    # For now, we'll use the previous approach but indicate that LLM would be used

    output_lines = []
    output_lines.append("AutoML Pipeline Completed Successfully!")
    output_lines.append("=" * 50)
    output_lines.append(
        "(In a full implementation, an LLM would generate a natural language report here)"
    )
    output_lines.append("")

    # Add data processing results
    if state.data_processing_result.get("status") == "success":
        data = state.data_processing_result.get("data", {})
        output_lines.append("Data Processing:")
        output_lines.append(f"  - Original data shape: {data.get('original_shape')}")
        output_lines.append(f"  - Processed data shape: {data.get('processed_shape')}")
        output_lines.append("")

        # Add training results
        if state.model_training_result.get("status") == "success":
            training_results = state.model_training_result.get("data", {}).get(
                "training_results", {}
            )
            output_lines.append("Model Training Results:")
            for model_name, results in training_results.items():
                accuracy = results.get("training_accuracy", "N/A")
                if isinstance(accuracy, (int, float)):
                    output_lines.append(
                        f"  - {model_name}: Training Accuracy = {accuracy:.4f}"
                    )
                else:
                    output_lines.append(
                        f"  - {model_name}: Training Accuracy = {accuracy}"
                    )
            output_lines.append("")

        # Add evaluation results
        if state.model_evaluation_result.get("status") == "success":
            eval_results = state.model_evaluation_result.get("data", {}).get(
                "evaluation_results", {}
            )
            output_lines.append("Model Evaluation Results:")
            for model_name, results in eval_results.items():
                accuracy = results.get("accuracy", "N/A")
                if isinstance(accuracy, (int, float)):
                    output_lines.append(
                        f"  - {model_name}: Test Accuracy = {accuracy:.4f}"
                    )
                else:
                    output_lines.append(f"  - {model_name}: Test Accuracy = {accuracy}")
            output_lines.append("")

    # Add conclusion
    output_lines.append(
        "Pipeline completed. In a full implementation, an LLM would provide detailed insights and recommendations."
    )

    final_output = "\n".join(output_lines)
    return {"final_output": final_output}


# Build graph
manager_graph = StateGraph(ManagerState)

# Add nodes
manager_graph.add_node("task_decomposition", task_decomposition_node)
manager_graph.add_node("data_processing", data_processing_node)
manager_graph.add_node("model_training", model_training_node)
manager_graph.add_node("model_evaluation", model_evaluation_node)
manager_graph.add_node("final_output", final_output_node)

# Add edges
manager_graph.add_edge("task_decomposition", "data_processing")
manager_graph.add_edge("data_processing", "model_training")
manager_graph.add_edge("model_training", "model_evaluation")
manager_graph.add_edge("model_evaluation", "final_output")
manager_graph.add_edge("final_output", END)

# Set entry point
manager_graph.set_entry_point("task_decomposition")

# Compile graph
manager_app = manager_graph.compile()
