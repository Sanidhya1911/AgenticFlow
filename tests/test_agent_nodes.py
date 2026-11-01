"""Unit tests for LangGraph agent execution nodes (agent_nodes.py)."""

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from langgraph.graph import StateGraph, START, END

from agent_nodes import executor_node, corrector_node
from graph_state import AgentState, create_initial_state
from schemas import APIPayload, EndpointContext
import llm_client


@pytest.fixture
def dummy_endpoint_context() -> dict:
    """Fixture providing dummy OpenAPI endpoint context dict."""
    return {
        "endpoint_path": "/orders",
        "method": "POST",
        "description": "Create an order",
        "required_params": [],
        "request_schema": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}, "quantity": {"type": "integer"}},
            "required": ["item_id"],
        },
        "base_url": "http://testserver",
    }


@pytest.fixture
def dummy_payload() -> dict:
    """Fixture providing dummy request payload dict."""
    return {
        "endpoint": "/orders",
        "method": "POST",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer token123"},
        "json_body": {"item_id": "item-abc", "quantity": 2},
        "query_params": {"notify": "true"},
    }


@pytest.mark.asyncio
async def test_executor_node_2xx_success(dummy_endpoint_context, dummy_payload):
    """Test executor_node successfully handles 2xx responses."""
    state: AgentState = {
        "endpoint_context": dummy_endpoint_context,
        "current_payload": dummy_payload,
        "attempt_count": 0,
        "last_response_status": 0,
        "last_error_message": "",
        "success": False,
    }

    mock_response = httpx.Response(
        status_code=200,
        json={"order_id": "ord-999", "status": "created"},
        request=httpx.Request("POST", "http://testserver/orders"),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_response

        updates = await executor_node(state)

        assert updates["success"] is True
        assert updates["last_response_status"] == 200
        # Verify request parameters dispatched
        mock_instance.request.assert_called_once_with(
            method="POST",
            url="/orders",
            headers={"Content-Type": "application/json", "Authorization": "Bearer token123"},
            params={"notify": "true"},
            json={"item_id": "item-abc", "quantity": 2},
        )


@pytest.mark.asyncio
async def test_executor_node_4xx_failure(dummy_endpoint_context, dummy_payload):
    """Test executor_node handles 4xx client errors, captures error body, and increments attempts."""
    state: AgentState = {
        "endpoint_context": dummy_endpoint_context,
        "current_payload": dummy_payload,
        "attempt_count": 1,
        "last_response_status": 0,
        "last_error_message": "",
        "success": False,
    }

    error_body = '{"detail": "Field item_id is invalid"}'
    mock_response = httpx.Response(
        status_code=422,
        text=error_body,
        request=httpx.Request("POST", "http://testserver/orders"),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_response

        updates = await executor_node(state)

        assert updates["success"] is False
        assert updates["last_response_status"] == 422
        assert updates["last_error_message"] == error_body
        assert updates["attempt_count"] == 2  # 1 + 1


@pytest.mark.asyncio
async def test_executor_node_5xx_failure(dummy_endpoint_context, dummy_payload):
    """Test executor_node handles 5xx server errors, capturing status and message."""
    state: AgentState = {
        "endpoint_context": dummy_endpoint_context,
        "current_payload": dummy_payload,
        "attempt_count": 0,
        "last_response_status": 0,
        "last_error_message": "",
        "success": False,
    }

    server_error_body = "Internal Server Error: Database unavailable"
    mock_response = httpx.Response(
        status_code=500,
        text=server_error_body,
        request=httpx.Request("POST", "http://testserver/orders"),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_response

        updates = await executor_node(state)

        assert updates["success"] is False
        assert updates["last_response_status"] == 500
        assert updates["last_error_message"] == server_error_body
        assert updates["attempt_count"] == 1


@pytest.mark.asyncio
async def test_executor_node_network_request_error(dummy_endpoint_context, dummy_payload):
    """Test executor_node catches network/connection exceptions gracefully."""
    state: AgentState = {
        "endpoint_context": dummy_endpoint_context,
        "current_payload": dummy_payload,
        "attempt_count": 0,
        "last_response_status": 0,
        "last_error_message": "",
        "success": False,
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.side_effect = httpx.ConnectTimeout("Connection timed out after 5.0s")

        updates = await executor_node(state)

        assert updates["success"] is False
        assert updates["last_response_status"] == 0
        assert "Connection timed out" in updates["last_error_message"]
        assert updates["attempt_count"] == 1


@pytest.mark.asyncio
async def test_executor_node_with_pydantic_models():
    """Test executor_node works seamlessly with Pydantic APIPayload and EndpointContext."""
    ctx = EndpointContext(
        endpoint_path="/health",
        method="GET",
        description="Health check",
    )
    payload = APIPayload(
        endpoint="/health",
        method="GET",
        headers={"Accept": "application/json"},
        json_body={},
    )
    state = create_initial_state(endpoint_context=ctx, initial_payload=payload)

    mock_response = httpx.Response(
        status_code=200,
        json={"status": "healthy"},
        request=httpx.Request("GET", "http://localhost:8000/health"),
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        mock_instance.request.return_value = mock_response

        updates = await executor_node(state)

        assert updates["success"] is True
        assert updates["last_response_status"] == 200
        mock_instance.request.assert_called_once_with(
            method="GET",
            url="/health",
            headers={"Accept": "application/json"},
        )


@pytest.mark.asyncio
async def test_corrector_node_default_mock():
    """Test corrector_node invokes llm_client.correct_payload and returns dict update."""
    state: AgentState = {
        "endpoint_context": {"endpoint_path": "/orders", "method": "POST"},
        "current_payload": {
            "endpoint": "/orders",
            "method": "POST",
            "json_body": {"item_id": "test"},
        },
        "last_error_message": '{"detail": "item not found"}',
        "attempt_count": 1,
        "last_response_status": 404,
        "success": False,
    }

    updates = await corrector_node(state)

    assert isinstance(updates, dict)
    assert "current_payload" in updates
    assert updates["current_payload"]["json_body"] == {"item_id": "test"}


@pytest.mark.asyncio
async def test_corrector_node_with_mocked_llm_correction():
    """Test corrector_node returns updated current_payload when correct_payload is mocked."""
    state: AgentState = {
        "endpoint_context": {"endpoint_path": "/users", "method": "POST"},
        "current_payload": {
            "endpoint": "/users",
            "method": "POST",
            "json_body": {"email": "invalid_email"},
        },
        "last_error_message": "Invalid email address format",
        "attempt_count": 1,
        "last_response_status": 422,
        "success": False,
    }

    corrected_payload_dict = {
        "endpoint": "/users",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json_body": {"email": "valid.user@example.com"},
    }

    with patch("llm_client.correct_payload", new_callable=AsyncMock) as mock_correct:
        mock_correct.return_value = corrected_payload_dict

        updates = await corrector_node(state)

        assert updates == {"current_payload": corrected_payload_dict}
        mock_correct.assert_called_once_with(state)


@pytest.mark.asyncio
async def test_langgraph_full_agent_loop_integration():
    """Test complete LangGraph StateGraph cycle combining executor_node and corrector_node."""
    # Build LangGraph StateGraph
    builder = StateGraph(AgentState)
    builder.add_node("executor", executor_node)
    builder.add_node("corrector", corrector_node)

    def route_after_executor(state: AgentState) -> str:
        if state.get("success"):
            return END
        if state.get("attempt_count", 0) >= state.get("max_retries", 3):
            return END
        return "corrector"

    builder.add_edge(START, "executor")
    builder.add_conditional_edges("executor", route_after_executor, ["corrector", END])
    builder.add_edge("corrector", "executor")

    graph = builder.compile()

    # Define mock responses: 1st attempt returns 422, 2nd attempt returns 200
    mock_422 = httpx.Response(
        status_code=422,
        text='{"detail": "Missing item_id"}',
        request=httpx.Request("POST", "http://testserver/orders"),
    )
    mock_200 = httpx.Response(
        status_code=200,
        json={"id": "ord-123", "status": "confirmed"},
        request=httpx.Request("POST", "http://testserver/orders"),
    )

    corrected_payload = {
        "endpoint": "/orders",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json_body": {"item_id": "valid-item-123"},
    }

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("llm_client.correct_payload", new_callable=AsyncMock) as mock_corrector:

        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_instance
        # 1st call fails with 422, 2nd call succeeds with 200
        mock_instance.request.side_effect = [mock_422, mock_200]
        mock_corrector.return_value = corrected_payload

        initial_state: AgentState = {
            "endpoint_context": {"endpoint_path": "/orders", "method": "POST"},
            "current_payload": {
                "endpoint": "/orders",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "json_body": {},  # Invalid, triggers 422
            },
            "attempt_count": 0,
            "last_response_status": 0,
            "last_error_message": "",
            "success": False,
            "max_retries": 3,
        }

        final_state = await graph.ainvoke(initial_state)

        # Verification of complete loop execution
        assert final_state["success"] is True
        assert final_state["last_response_status"] == 200
        assert final_state["attempt_count"] == 1  # Incremented on the 422 attempt
        assert final_state["current_payload"]["json_body"] == {"item_id": "valid-item-123"}
        assert mock_instance.request.call_count == 2
        mock_corrector.assert_called_once()
