"""Tests for Pydantic V2 structured output models and agent handoffs."""

import pytest
from pydantic import ValidationError

from schemas import (
    APIPayload,
    APIResponse,
    EndpointContext,
    TestPlanStep,
    CorrectionFeedback,
    AgentHandoff,
)
from openapi_parser import extract_endpoints
from pathlib import Path

SAMPLE_API_PATH = Path(__file__).parent.parent / "sample_api.json"


def test_api_payload_instantiation_and_validation():
    """Verify APIPayload instantiation, normalization, and field descriptions."""
    payload = APIPayload(
        endpoint="/orders/123e4567-e89b-12d3-a456-426614174000",
        method="post",  # should be normalized to uppercase
        headers={"Content-Type": "application/json", "Authorization": "Bearer test-token"},
        json_body={"customer_id": "cust-001", "items": [{"quantity": 2}]},
    )

    assert payload.method == "POST"
    assert payload.endpoint == "/orders/123e4567-e89b-12d3-a456-426614174000"
    assert payload.headers["Authorization"] == "Bearer test-token"
    assert payload.json_body["customer_id"] == "cust-001"

    # Verify JSON Schema descriptions for LLM prompting
    json_schema = APIPayload.model_json_schema()
    assert "properties" in json_schema
    props = json_schema["properties"]

    assert "endpoint" in props
    assert "description" in props["endpoint"]
    assert len(props["endpoint"]["description"]) > 20

    assert "method" in props
    assert "description" in props["method"]

    assert "headers" in props
    assert "description" in props["headers"]

    assert "json_body" in props
    assert "description" in props["json_body"]


def test_api_payload_invalid_method():
    """Verify validation error on invalid HTTP method verb."""
    with pytest.raises(ValidationError):
        APIPayload(
            endpoint="/orders",
            method="INVALID_VERB",
            headers={},
            json_body={},
        )


def test_api_response_model():
    """Verify APIResponse fields, auto is_success flag, and schema descriptions."""
    # 201 Created -> is_success should be True
    resp_201 = APIResponse(
        status_code=201,
        response_data={"order_id": "ord-123", "status": "pending"},
        response_time_ms=78.5,
    )
    assert resp_201.is_success is True
    assert resp_201.response_time_ms == 78.5
    assert resp_201.error_message is None

    # 400 Bad Request -> is_success should be False
    resp_400 = APIResponse(
        status_code=400,
        response_data={"code": "INVALID_ITEM", "message": "Item SKU not found"},
        response_time_ms=32.1,
    )
    assert resp_400.is_success is False

    # Check JSON Schema descriptions
    json_schema = APIResponse.model_json_schema()
    props = json_schema["properties"]
    assert "status_code" in props
    assert "description" in props["status_code"]
    assert "response_data" in props
    assert "description" in props["response_data"]
    assert "response_time_ms" in props
    assert "description" in props["response_time_ms"]


def test_endpoint_context_from_openapi_operation():
    """Verify EndpointContext creation and factory method with OpenAPI parser output."""
    endpoints = extract_endpoints(SAMPLE_API_PATH)
    post_orders_op = next(ep for ep in endpoints if ep.path == "/orders" and ep.method == "POST")

    context = EndpointContext.from_endpoint_operation(post_orders_op)

    assert context.endpoint_path == "/orders"
    assert context.method == "POST"
    assert "Create a new order" in context.description
    assert isinstance(context.required_params, list)
    assert isinstance(context.request_schema, dict)
    assert "customer_id" in context.request_schema.get("properties", {})

    # Check JSON Schema descriptions
    json_schema = EndpointContext.model_json_schema()
    props = json_schema["properties"]
    assert "endpoint_path" in props
    assert "description" in props["endpoint_path"]
    assert "required_params" in props
    assert "description" in props["required_params"]
    assert "description" in props["description"]


def test_agent_handoff_lifecycle():
    """Verify LangGraph deterministic handoff lifecycle models."""
    context = EndpointContext(
        endpoint_path="/orders",
        method="POST",
        description="Create order endpoint",
        required_params=[],
    )

    payload = APIPayload(
        endpoint="/orders",
        method="POST",
        headers={"Content-Type": "application/json"},
        json_body={"item_id": "abc"},
    )

    step = TestPlanStep(
        step_id=1,
        title="Create Order Initial",
        context=context,
        payload=payload,
        expected_status=201,
    )

    handoff = AgentHandoff(
        current_agent="planner",
        step=step,
    )

    assert handoff.current_agent == "planner"
    assert handoff.retry_count == 0

    # Executor execution
    handoff.current_agent = "executor"
    handoff.response = APIResponse(
        status_code=400,
        response_data={"error": "Missing quantity"},
        response_time_ms=54.2,
    )
    assert handoff.response.status_code == 400

    # Corrector feedback
    handoff.current_agent = "corrector"
    handoff.feedback = CorrectionFeedback(
        error_type="schema_mismatch",
        diagnosis="API returned 400 because 'quantity' was missing from payload.",
        suggested_fix="Add quantity: 1 to the items array.",
        corrected_payload=APIPayload(
            endpoint="/orders",
            method="POST",
            headers={"Content-Type": "application/json"},
            json_body={"item_id": "abc", "quantity": 1},
        ),
    )
    handoff.retry_count += 1

    assert handoff.retry_count == 1
    assert handoff.feedback.error_type == "schema_mismatch"
    assert handoff.feedback.corrected_payload.json_body["quantity"] == 1
