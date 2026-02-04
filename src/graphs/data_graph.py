"""
Data Processing Graph for AutoML System
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Annotated
import operator

from ..agents.data_agent import DataAgent


# Define state schema
class DataState(BaseModel):
    task: Dict[str, Any] = Field(default={})
    result: Dict[str, Any] = Field(default={})
    messages: Annotated[list, operator.add] = Field(default=[])


# Initialize agent
data_agent = DataAgent()


# Define nodes
def process_data_node(state: DataState) -> Dict[str, Any]:
    """Process data using DataAgent"""
    result = data_agent.process_task(state.task)
    return {"result": result}


def router(state: DataState) -> str:
    """Router to determine next step"""
    if state.result.get("status") == "success":
        return "success"
    else:
        return "error"


# Build graph
data_graph = StateGraph(DataState)

# Add nodes
data_graph.add_node("process_data", process_data_node)

# Add edges
data_graph.add_edge("process_data", END)

# Set entry point
data_graph.set_entry_point("process_data")

# Compile graph
data_app = data_graph.compile()
