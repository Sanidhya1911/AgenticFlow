"""Unit tests for LangGraph state schema (graph_state.py)."""

import pytest
from pydantic import ValidationError
from langgraph.graph import StateGraph, START, END

from graph_state import (
    AgentState,
    AgentStateModel,
    create_initial_state,
)
from schemas import APIPayload, EndpointContext


@pytest.fixture
def dummy_endpoint_context() -> EndpointContext:
    """Fixture providing dummy EndpointContext."""
    return EndpointContext(
        endpoint_path="/orders",
        method="POST",
        description="Create a new customer order",
        required_params=[],
        request_schema={
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        },
    )


def test_agent_state_keys_and_types():
    """Verify AgentState TypedDict contains all required keys."""
    state: AgentState = {
        "endpoint_context": {
            "endpoint_path": "/orders",
            "method": "POST",
            "description": "Create an order",
            "required_params": [],
            "request_schema": {},
        },
        "current_payload": {
            "endpoint": "/orders",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "json_body": {"item_id": "item-123"},
        },
        "attempt_count": 1,
        "last_response_status": 400,
        "last_error_message": '{"detail": "item-123 is out of stock"}',
        "success": False,
    }

    assert state["endpoint_context"]["endpoint_path"] == "/orders"
    assert state["current_payload"]["method"] == "POST"
    assert state["attempt_count"] == 1
    assert state["last_response_status"] == 400
    assert "out of stock" in state["last_error_message"]
    assert state["success"] is False


def test_create_initial_state_from_models(dummy_endpoint_context):
    """Verify create_initial_state initializes default state correctly from Pydantic models."""
    initial_payload = APIPayload(
        endpoint="/orders",
        method="POST",
        headers={"Content-Type": "application/json"},
        json_body={"item_id": "test-item"},
    )

    state = create_initial_state(
        endpoint_context=dummy_endpoint_context,
        initial_payload=initial_payload,
        max_retries=5,
    )

    assert state["endpoint_context"]["endpoint_path"] == "/orders"
    assert state["current_payload"]["json_body"]["item_id"] == "test-item"
    assert state["attempt_count"] == 0
    assert state["last_response_status"] == 0
    assert state["last_error_message"] == ""
    assert state["success"] is False
    assert state["max_retries"] == 5
    assert state["history"] == []


def test_create_initial_state_from_dicts():
    """Verify create_initial_state works seamlessly with plain dictionaries."""
    ctx_dict = {
        "endpoint_path": "/users",
        "method": "GET",
        "description": "List users",
        "required_params": [],
        "request_schema": {},
    }

    state = create_initial_state(endpoint_context=ctx_dict)

    assert state["endpoint_context"]["endpoint_path"] == "/users"
    assert state["current_payload"] == {}
    assert state["attempt_count"] == 0
    assert state["success"] is False


def test_agent_state_model_validation():
    """Verify AgentStateModel validates fields and serializes to TypedDict."""
    model = AgentStateModel(
        endpoint_context={"endpoint_path": "/orders"},
        current_payload={"endpoint": "/orders", "method": "GET"},
        attempt_count=2,
        last_response_status=200,
        last_error_message="",
        success=True,
    )

    assert model.success is True
    assert model.attempt_count == 2

    state_dict = model.to_state_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["last_response_status"] == 200
    assert state_dict["success"] is True


def test_multi_agent_execution_loop_state_progression(dummy_endpoint_context):
    """Simulate multi-agent lifecycle state mutations across Planner, Executor, and Corrector."""
    # 1. Planner initialization
    state = create_initial_state(dummy_endpoint_context)
    state["current_payload"] = {
        "endpoint": "/orders",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json_body": {},  # deliberately empty, will fail validation
    }

    # 2. Executor run (fails with 422)
    state["attempt_count"] += 1
    state["last_response_status"] = 422
    state["last_error_message"] = "Field 'item_id' is required"
    state["success"] = False

    assert state["attempt_count"] == 1
    assert state["success"] is False

    # 3. Corrector updates payload
    state["current_payload"]["json_body"] = {"item_id": "item-abc-123"}

    # 4. Executor retry run (succeeds with 201)
    state["attempt_count"] += 1
    state["last_response_status"] = 201
    state["last_error_message"] = ""
    state["success"] = True

    assert state["attempt_count"] == 2
    assert state["last_response_status"] == 201
    assert state["success"] is True


def test_langgraph_integration_with_agent_state(dummy_endpoint_context):
    """Directly compile a LangGraph StateGraph utilizing AgentState and execute a step."""
    def mock_planner_node(state: AgentState) -> dict:
        return {
            "current_payload": {
                "endpoint": state["endpoint_context"]["endpoint_path"],
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "json_body": {"item_id": "item-langgraph"},
            },
            "attempt_count": state["attempt_count"] + 1,
        }

    def mock_executor_node(state: AgentState) -> dict:
        return {
            "last_response_status": 201,
            "success": True,
            "last_error_message": "",
        }

    # Build LangGraph StateGraph using AgentState schema
    builder = StateGraph(AgentState)
    builder.add_node("planner", mock_planner_node)
    builder.add_node("executor", mock_executor_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", END)

    compiled_graph = builder.compile()

    init_state = create_initial_state(dummy_endpoint_context)
    final_output = compiled_graph.invoke(init_state)

    assert final_output["attempt_count"] == 1
    assert final_output["current_payload"]["json_body"]["item_id"] == "item-langgraph"
    assert final_output["last_response_status"] == 201
    assert final_output["success"] is True
