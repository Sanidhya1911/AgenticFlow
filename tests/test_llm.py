"""Async unit tests for Groq LLM client and structured APIPayload generation."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from groq import APIError, RateLimitError
import httpx

from llm_client import (
    generate_api_payload,
    LLMClientError,
    LLMRateLimitError,
    LLMValidationError,
)
from schemas import APIPayload, EndpointContext


@pytest.fixture
def sample_context() -> EndpointContext:
    """Fixture providing an EndpointContext for testing."""
    return EndpointContext(
        endpoint_path="/orders/{orderId}",
        method="GET",
        description="Retrieve an existing order by ID",
        required_params=[
            {
                "name": "orderId",
                "in": "path",
                "required": True,
                "type": "string",
                "description": "UUID of the order",
            }
        ],
        request_schema={},
    )


def create_mock_completion(arguments_dict: dict, as_tool_call: bool = True) -> MagicMock:
    """Helper to create a mocked Groq ChatCompletion structure."""
    mock_choice = MagicMock()
    mock_message = MagicMock()
    raw_json = json.dumps(arguments_dict)

    if as_tool_call:
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_api_payload"
        mock_tool_call.function.arguments = raw_json
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None
    else:
        mock_message.tool_calls = None
        mock_message.content = raw_json

    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


@pytest.mark.asyncio
async def test_generate_api_payload_tool_call_success(sample_context):
    """Verify successful generation and parsing of APIPayload via tool_call."""
    payload_data = {
        "endpoint": "/orders/123e4567-e89b-12d3-a456-426614174000",
        "method": "GET",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer token123"},
        "json_body": {},
        "query_params": {"include_items": True},
    }

    mock_client = MagicMock()
    mock_completion = create_mock_completion(payload_data, as_tool_call=True)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    result = await generate_api_payload(sample_context, client=mock_client)

    assert isinstance(result, APIPayload)
    assert result.endpoint == "/orders/123e4567-e89b-12d3-a456-426614174000"
    assert result.method == "GET"
    assert result.headers["Authorization"] == "Bearer token123"
    assert result.query_params["include_items"] is True
    assert result.json_body == {}
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_generate_api_payload_json_content_success(sample_context):
    """Verify fallback to direct JSON message content when tool_calls is absent."""
    payload_data = {
        "endpoint": "/orders/abc-987",
        "method": "DELETE",
        "headers": {"Content-Type": "application/json"},
        "json_body": {},
    }

    mock_client = MagicMock()
    mock_completion = create_mock_completion(payload_data, as_tool_call=False)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    result = await generate_api_payload(sample_context, client=mock_client)

    assert isinstance(result, APIPayload)
    assert result.endpoint == "/orders/abc-987"
    assert result.method == "DELETE"


@pytest.mark.asyncio
async def test_generate_api_payload_rate_limit_retry(sample_context):
    """Verify exponential backoff retry when Groq returns RateLimitError."""
    payload_data = {
        "endpoint": "/orders",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json_body": {"customer_id": "cust-01"},
    }

    mock_client = MagicMock()
    mock_success = create_mock_completion(payload_data)

    dummy_response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    rate_limit_err = RateLimitError("Rate limit reached", response=dummy_response, body=None)

    # First attempt fails with RateLimitError, second succeeds
    mock_client.chat.completions.create = AsyncMock(side_effect=[rate_limit_err, mock_success])

    result = await generate_api_payload(
        sample_context,
        client=mock_client,
        max_retries=2,
        initial_delay=0.01,  # Short delay for fast test
    )

    assert isinstance(result, APIPayload)
    assert result.method == "POST"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_api_payload_rate_limit_exhausted(sample_context):
    """Verify LLMRateLimitError raised when rate limits persist past max_retries."""
    mock_client = MagicMock()
    dummy_response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    rate_limit_err = RateLimitError("Rate limit reached", response=dummy_response, body=None)
    mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

    with pytest.raises(LLMRateLimitError) as exc_info:
        await generate_api_payload(
            sample_context,
            client=mock_client,
            max_retries=2,
            initial_delay=0.01,
        )

    assert "Groq RateLimitError after 2 attempts" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_api_payload_malformed_json_recovery(sample_context):
    """Verify self-correction when LLM returns invalid JSON on first attempt."""
    mock_client = MagicMock()

    # Attempt 1: Malformed JSON string
    bad_choice = MagicMock()
    bad_tool = MagicMock()
    bad_tool.function.arguments = "{ not a valid json string"
    bad_choice.message.tool_calls = [bad_tool]
    bad_choice.message.content = None
    bad_completion = MagicMock()
    bad_completion.choices = [bad_choice]

    # Attempt 2: Valid payload
    good_completion = create_mock_completion({
        "endpoint": "/orders/123",
        "method": "GET",
        "headers": {},
        "json_body": {},
    })

    mock_client.chat.completions.create = AsyncMock(side_effect=[bad_completion, good_completion])

    result = await generate_api_payload(
        sample_context,
        client=mock_client,
        max_retries=2,
        initial_delay=0.01,
    )

    assert isinstance(result, APIPayload)
    assert result.endpoint == "/orders/123"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_api_payload_validation_failure_exhausted(sample_context):
    """Verify LLMValidationError raised when output fails Pydantic schema validation across all retries."""
    mock_client = MagicMock()

    # Payload with invalid HTTP method to trigger Pydantic ValidationError
    invalid_payload = {
        "endpoint": "/orders",
        "method": "INVALID_VERB",
        "headers": {},
        "json_body": {},
    }
    invalid_completion = create_mock_completion(invalid_payload)
    mock_client.chat.completions.create = AsyncMock(return_value=invalid_completion)

    with pytest.raises(LLMValidationError) as exc_info:
        await generate_api_payload(
            sample_context,
            client=mock_client,
            max_retries=2,
            initial_delay=0.01,
        )

    assert "Failed to generate valid APIPayload after 2 attempts" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_api_payload_api_error(sample_context):
    """Verify LLMClientError raised on fatal Groq APIError."""
    mock_client = MagicMock()
    dummy_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    auth_err = APIError("Invalid API Key", request=dummy_request, body=None)
    mock_client.chat.completions.create = AsyncMock(side_effect=auth_err)

    with pytest.raises(LLMClientError) as exc_info:
        await generate_api_payload(
            sample_context,
            client=mock_client,
            max_retries=2,
            initial_delay=0.01,
        )

    assert "Groq API error encountered" in str(exc_info.value)
