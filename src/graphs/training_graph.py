"""
Training Graph for AutoML System
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Annotated
import operator

from ..agents.training_agent import TrainingAgent


# Define state schema
class TrainingState(BaseModel):
    task: Dict[str, Any] = Field(default={})
    result: Dict[str, Any] = Field(default={})
    messages: Annotated[list, operator.add] = Field(default=[])


# Initialize agent
training_agent = TrainingAgent()


# Define nodes
def train_model_node(state: TrainingState) -> Dict[str, Any]:
    """Train models using TrainingAgent"""
    result = training_agent.process_task(state.task)
    return {"result": result}


# Build graph
training_graph = StateGraph(TrainingState)

# Add nodes
training_graph.add_node("train_model", train_model_node)

# Add edges
training_graph.add_edge("train_model", END)

# Set entry point
training_graph.set_entry_point("train_model")

# Compile graph
training_app = training_graph.compile()
