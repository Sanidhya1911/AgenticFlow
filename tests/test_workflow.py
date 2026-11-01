"""Unit tests for LangGraph compiled execution loop (workflow.py)."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from langgraph.graph import END

from graph_state import AgentState, create_initial_state
from workflow import (
    app,
    route_after_execution,
    run_workflow,
    workflow,
)


def test_route_after_execution_success():
    """Verify route_after_execution routes to END when success is True."""
    state_success_first_attempt: AgentState = {
        "success": True,
        "attempt_count": 1,
    }
    assert route_after_execution(state_success_first_attempt) == END

    state_success_multiple_attempts: AgentState = {
        "success": True,
        "attempt_count": 2,
    }
    assert route_after_execution(state_success_multiple_attempts) == END


def test_route_after_execution_failure_under_retry_limit():
    """Verify route_after_execution routes to corrector_node when success is False and attempt_count < 3."""
    state_fail_0: AgentState = {
        "success": False,
        "attempt_count": 0,
    }
    assert route_after_execution(state_fail_0) == "corrector_node"

    state_fail_1: AgentState = {
        "success": False,
        "attempt_count": 1,
    }
    assert route_after_execution(state_fail_1) == "corrector_node"

    state_fail_2: AgentState = {
        "success": False,
        "attempt_count": 2,
    }
    assert route_after_execution(state_fail_2) == "corrector_node"


def test_route_after_execution_max_retries_reached():
    """Verify route_after_execution routes to END when attempt_count >= 3."""
    state_max_3: AgentState = {
        "success": False,
        "attempt_count": 3,
    }
    assert route_after_execution(state_max_3) == END

    state_exceeded_4: AgentState = {
        "success": False,
        "attempt_count": 4,
    }
    assert route_after_execution(state_exceeded_4) == END


def test_workflow_graph_structure():
    """Verify nodes and structure of compiled workflow."""
    assert "executor_node" in workflow.nodes
    assert "corrector_node" in workflow.nodes
    assert app is not None


@pytest.mark.asyncio
async def test_run_workflow_direct_success():
    """Test run_workflow terminates on first attempt when API returns 200."""
    mock_200 = httpx.Response(
        status_code=200,
        json={"status": "ok"},
        request=httpx.Request("GET", "http://localhost:8000/health"),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_200

        result = await run_workflow()

        assert result["success"] is True
        assert result["last_response_status"] == 200
        assert mock_instance.request.call_count == 1


@pytest.mark.asyncio
async def test_run_workflow_self_healing_success():
    """Test run_workflow executes self-healing loop: 1st call fails 422, corrected payload succeeds 200."""
    mock_422 = httpx.Response(
        status_code=422,
        text='{"detail": "Missing property \'name\'"}',
        request=httpx.Request("POST", "http://testserver/items"),
    )
    mock_200 = httpx.Response(
        status_code=200,
        json={"id": "item-1", "name": "Item Name"},
        request=httpx.Request("POST", "http://testserver/items"),
    )

    corrected_payload = {
        "endpoint": "/items",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json_body": {"name": "Item Name"},
    }

    initial_state = create_initial_state(
        endpoint_context={
            "endpoint_path": "/items",
            "method": "POST",
            "base_url": "http://testserver",
        },
        initial_payload={
            "endpoint": "/items",
            "method": "POST",
            "json_body": {},
        },
    )

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("agent_nodes.correct_payload", new_callable=AsyncMock) as mock_corrector:

        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.side_effect = [mock_422, mock_200]
        mock_corrector.return_value = corrected_payload

        final_state = await run_workflow(initial_state)

        assert final_state["success"] is True
        assert final_state["last_response_status"] == 200
        assert final_state["attempt_count"] == 1
        assert final_state["current_payload"]["json_body"] == {"name": "Item Name"}
        assert mock_instance.request.call_count == 2
        mock_corrector.assert_called_once()


@pytest.mark.asyncio
async def test_run_workflow_exhausts_retries():
    """Test run_workflow terminates after 3 failed attempts (reaches max retries)."""
    mock_500 = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("GET", "http://testserver/fail"),
    )

    initial_state = create_initial_state(
        endpoint_context={
            "endpoint_path": "/fail",
            "method": "GET",
            "base_url": "http://testserver",
        },
        initial_payload={"endpoint": "/fail", "method": "GET"},
    )

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("agent_nodes.correct_payload", new_callable=AsyncMock) as mock_corrector:

        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_500
        mock_corrector.return_value = {"endpoint": "/fail", "method": "GET"}

        final_state = await run_workflow(initial_state)

        assert final_state["success"] is False
        assert final_state["attempt_count"] == 3
        assert final_state["last_response_status"] == 500
        assert mock_instance.request.call_count == 3
        assert mock_corrector.call_count == 2

