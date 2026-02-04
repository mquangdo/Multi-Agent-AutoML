"""
Example script to run the full AutoML pipeline with iris dataset
This version fixes import issues by using absolute imports
"""

import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)


def run_iris_automl():
    """Run AutoML pipeline with iris dataset"""
    print("Running AutoML pipeline with Iris dataset")
    print("=" * 50)

    # Import using absolute imports
    from src.graphs.manager_graph import manager_app

    # Example request for iris dataset (using English to avoid encoding issues)
    user_request = "This is the path to the data file data/iris_data.csv. The task is to classify iris flower species. Please help me train SVM and report the results."

    print(f"User request: {user_request}")
    print("=" * 50)

    # Create initial state
    initial_state = {"user_input": user_request}

    try:
        # Run the manager app
        result = manager_app.invoke(initial_state)

        # Print final output
        print("\nFINAL RESULTS:")
        print("=" * 50)
        print(result["final_output"])

        return result
    except Exception as e:
        print(f"Error running AutoML pipeline: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = run_iris_automl()
