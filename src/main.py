"""
Main entry point for AutoML Agent System
Allows users to input requests from the terminal
"""

import sys
import os
from opik import configure 
from opik.integrations.langchain import OpikTracer 

configure() 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

##Opik Tracer setup
project_name = 'AutoML-Agent'

def run_automl_pipeline(user_input: str):
    """
    Run the complete AutoML pipeline with user input

    Args:
        user_input (str): User's request for AutoML pipeline
    """
    print(f"Running AutoML pipeline for request: {user_input}")
    print("=" * 50)

    try:
        # Import here to avoid import issues
        from src.graphs.manager_graph import manager_app

        # Create initial state
        initial_state = {"user_input": user_input}

        # Run the manager app
        tracer = OpikTracer(graph=manager_app.get_graph(xray=True), project_name=project_name) 

        result = manager_app.invoke(initial_state, config={"callbacks": [tracer]})

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


def run_interactive_mode():
    """Run AutoML in interactive mode allowing user to input requests"""
    print("AutoML System - Interactive Mode")
    print("=" * 50)
    print("Enter your machine learning request or 'quit' to exit.")
    print(
        "Example: This is the path to the data file data/iris_data.csv. The task is to classify iris flower species. Please help me train SVM and Random Forest models and report the results."
    )
    print("=" * 50)

    while True:
        try:
            # Get user input
            user_request = input("\nEnter your request: ").strip()

            # Check if user wants to quit
            if user_request.lower() in ["quit", "exit", "q"]:
                print("Thank you for using AutoML System!")
                break

            # Check if user entered something
            if not user_request:
                print("Please enter a valid request or 'quit' to exit.")
                continue

            # Run the pipeline with user request
            run_automl_pipeline(user_request)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except EOFError:
            print("\n\nEnd of input. Exiting...")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode - use provided arguments as request
        user_request = " ".join(sys.argv[1:])
        run_automl_pipeline(user_request)
    else:
        # Interactive mode - prompt user for input
        run_interactive_mode()
