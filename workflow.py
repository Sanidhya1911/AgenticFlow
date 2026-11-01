"""LangGraph execution workflow for AgenticFlow.

This module compiles the multi-agent execution loop orchestrating autonomous
REST API testing and self-healing error remediation.

Architecture:
- `executor_node`: Asynchronously executes HTTP requests via httpx.
- `route_after_execution`: Evaluates execution telemetry to direct control flow:
    - Route to `END` on success (HTTP 2xx).
    - Route to `corrector_node` on failure if attempt_count < 3.
    - Route to `END` if maximum retries (attempt_count >= 3) are reached.
- `corrector_node`: Remediates payloads using Groq Llama 3.3 70B and loops back
  to `executor_node`.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from langgraph.graph import StateGraph, START, END

from agent_nodes import executor_node, corrector_node
from graph_state import AgentState, create_initial_state

logger = logging.getLogger(__name__)


def route_after_execution(state: AgentState) -> str:
    """Conditionally route execution following the executor node attempt.

    Routing rules:
    - If state['success'] == True: route to END (successful API execution).
    - If state['success'] == False and state['attempt_count'] < 3: route to corrector_node (self-healing retry).
    - If state['attempt_count'] >= 3: route to END (maximum retry threshold reached).

    Args:
        state: The current AgentState containing execution metrics and results.

    Returns:
        str: Next node identifier ("corrector_node" or END).
    """
    if state.get("success") is True or state.get("success") == True:
        return END

    if not state.get("success", False) and state.get("attempt_count", 0) < 3:
        return "corrector_node"

    if state.get("attempt_count", 0) >= 3:
        return END

    return END


# 1. Initialize StateGraph with AgentState
workflow = StateGraph(AgentState)

# 2. Add worker nodes to the graph
workflow.add_node("executor_node", executor_node)
workflow.add_node("corrector_node", corrector_node)

# 3. Define graph edges and conditional routing
workflow.add_edge(START, "executor_node")
workflow.add_conditional_edges("executor_node", route_after_execution)
workflow.add_edge("corrector_node", "executor_node")

# 4. Compile graph into an executable application
app = workflow.compile()


async def run_workflow(
    initial_state: Optional[Union[AgentState, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute the compiled LangGraph self-healing REST API testing workflow.

    Args:
        initial_state: Optional initial AgentState or dictionary. If omitted,
            a default probe state is created.

    Returns:
        Dict[str, Any]: Final evaluated state dictionary after execution terminates.
    """
    if initial_state is None:
        initial_state = create_initial_state(
            endpoint_context={
                "endpoint_path": "/health",
                "method": "GET",
                "description": "Health check endpoint probe",
                "base_url": "http://localhost:8000",
            },
            initial_payload={
                "endpoint": "/health",
                "method": "GET",
            },
            max_retries=3,
        )

    logger.info("Executing AgenticFlow workflow with state: %s", initial_state)
    final_state = await app.ainvoke(initial_state)
    logger.info(
        "AgenticFlow workflow finished. Success=%s, Attempts=%s, Status=%s",
        final_state.get("success"),
        final_state.get("attempt_count"),
        final_state.get("last_response_status"),
    )
    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_workflow())
    print("\n--- Final Workflow State ---")
    for key, val in result.items():
        print(f"{key}: {val}")

