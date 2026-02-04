"""
Evaluation Graph for AutoML System
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Annotated
import operator

from ..agents.eval_agent import EvaluationAgent


# Define state schema
class EvaluationState(BaseModel):
    task: Dict[str, Any] = Field(default={})
    result: Dict[str, Any] = Field(default={})
    messages: Annotated[list, operator.add] = Field(default=[])


# Initialize agent
eval_agent = EvaluationAgent()


# Define nodes
def evaluate_model_node(state: EvaluationState) -> Dict[str, Any]:
    """Evaluate models using EvaluationAgent"""
    result = eval_agent.process_task(state.task)
    return {"result": result}


# Build graph
eval_graph = StateGraph(EvaluationState)

# Add nodes
eval_graph.add_node("evaluate_model", evaluate_model_node)

# Add edges
eval_graph.add_edge("evaluate_model", END)

# Set entry point
eval_graph.set_entry_point("evaluate_model")

# Compile graph
eval_app = eval_graph.compile()
