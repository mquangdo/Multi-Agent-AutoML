# AutoML Agent System

A multi-agent system for automated machine learning built with LangGraph and LangChain, now integrated with Large Language Models (LLMs).

## System Architecture

The system consists of a Manager Agent that orchestrates three specialized sub-agents:

1. **Data Agent**: Handles data preprocessing tasks with LLM assistance
2. **Training Agent**: Handles model training tasks with LLM assistance
3. **Evaluation Agent**: Handles model evaluation tasks with LLM assistance

## LLM Integration

The system now incorporates LLMs in several key areas:

1. **Task Decomposition**: LLM analyzes user requests and extracts key information
2. **Data Preprocessing**: LLM analyzes data characteristics and suggests preprocessing steps
3. **Hyperparameter Tuning**: LLM suggests optimal hyperparameters based on data characteristics
4. **Result Interpretation**: LLM generates natural language insights from evaluation results

## Features

- Task decomposition and delegation with LLM reasoning
- Modular agent design with LangGraph
- Support for multiple ML models (SVM, Random Forest)
- Complete AutoML pipeline from data to evaluation
- Natural language insights generation with LLM

## Requirements

- Python 3.8+
- LangChain
- LangGraph
- Scikit-learn
- Pandas
- NumPy

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Update the LLM configuration in `src/config/llm_config.py`:
- Choose provider (OpenAI or Ollama for local models)
- Select model (gpt-4, llama3, etc.)
- Adjust temperature and other parameters

## Usage

### Method 1: Simple Example (Recommended for testing)

For a quick test with your iris dataset:

```bash
cd automl-agent
python src/simple_main.py
```

Or use the example runner:

```bash
cd automl-agent
python src/run_example_simple.py
```

This will run the AutoML pipeline on your `data/iris_data.csv` file.

### Method 2: Full LLM Integration (Requires proper LLM setup)

Run the AutoML pipeline with a user request:

```bash
python src/run_example.py
```

With a specific request:

```bash
python src/run_example.py "Đây là đường dẫn đến dữ liệu data/iris_data.csv, bài toán là phân loại các loài hoa iris. Giúp tôi huấn luyện các mô hình SVM, Random forest và báo cáo lại cho tôi kết quả"
```

*Note: This method requires all LangChain/LangGraph dependencies to be installed and properly configured.*

## System Workflow

1. **Task Decomposition**: Manager Agent uses LLM to parse user request and decompose it into subtasks
2. **Data Processing**: Data Agent uses LLM to analyze data and suggest preprocessing steps
3. **Model Training**: Training Agent uses LLM to suggest optimal hyperparameters
4. **Model Evaluation**: Evaluation Agent uses LLM to generate insights from evaluation results
5. **Result Reporting**: Manager Agent aggregates results and uses LLM to generate natural language report

## Project Structure

```
automl-agent/
├── src/
│   ├── agents/           # Individual agent implementations with LLM integration
│   ├── graphs/           # LangGraph implementations for each agent
│   ├── utils/            # Utility functions
│   ├── config/           # Configuration files including LLM config
│   ├── main.py          # Main entry point (full LLM version)
│   ├── simple_main.py   # Simplified version for testing
│   ├── run_example.py   # Example runner (LLM version)
│   └── run_example_simple.py   # Example runner (simple version)
├── data/                # Data files (including your iris_data.csv)
├── requirements.txt     # Dependencies
├── README.md            # This file
└── EXAMPLE_USAGE.md     # Detailed usage instructions
```