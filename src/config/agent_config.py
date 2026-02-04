"""
Agent configuration settings
"""

AGENT_CONFIG = {
    "manager": {
        "name": "Manager Agent",
        "description": "Orchestrates the AutoML pipeline by delegating tasks to sub-agents",
    },
    "data": {"name": "Data Agent", "description": "Handles data preprocessing tasks"},
    "training": {
        "name": "Training Agent",
        "description": "Handles model training tasks",
    },
    "eval": {
        "name": "Evaluation Agent",
        "description": "Handles model evaluation tasks",
    },
}
