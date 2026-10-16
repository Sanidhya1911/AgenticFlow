"""Asynchronous Groq LLM client for AgenticFlow.

Utilizes Llama 3.3 70B via the official Groq Python SDK to generate structured,
deterministic APIPayload instances from OpenAPI EndpointContext definitions.
Features robust exponential-backoff retry logic for rate limits, API errors,
and Pydantic validation failures.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from groq import APIError, AsyncGroq, RateLimitError
from pydantic import ValidationError

from schemas import APIPayload, EndpointContext

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class LLMClientError(Exception):
    """Base exception for LLM client execution errors."""


class LLMValidationError(LLMClientError):
    """Raised when LLM output fails schema validation after retries."""


class LLMRateLimitError(LLMClientError):
    """Raised when Groq rate limits are exceeded and retries are exhausted."""


def get_api_payload_tool_definition() -> Dict[str, Any]:
    """Generates the Groq/OpenAI compatible tool schema for APIPayload."""
    schema = APIPayload.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": "submit_api_payload",
            "description": (
                "Submit the generated, concrete REST API request payload adhering strictly "
                "to the OpenAPI schema and business requirements."
            ),
            "parameters": schema,
        },
    }


def build_system_prompt() -> str:
    """Builds the comprehensive system prompt instructing the LLM as an API Test Engineer."""
    return (
        "You are an expert QA and API Test Automation Engineer working in the AgenticFlow framework.\n"
        "Your task is to generate a concrete, highly accurate REST API test request for a given target endpoint.\n\n"
        "Guidelines:\n"
        "1. Populate all path parameters with realistic dummy values matching the format/type specified "
        "(e.g., replace '{orderId}' with a valid UUID or ID). Never leave literal brackets in the endpoint URL.\n"
        "2. Ensure the HTTP method matches the operation (GET, POST, PUT, DELETE, PATCH, etc.).\n"
        "3. For requests with request bodies (POST, PUT, PATCH), construct a valid 'json_body' that fulfills "
        "all required fields, types, and constraints defined in the OpenAPI schema.\n"
        "4. For requests without a body (GET, DELETE), provide an empty dictionary {} for 'json_body'.\n"
        "5. Include relevant query parameters if required or useful for testing pagination/filtering.\n"
        "6. Provide standard headers, including 'Content-Type': 'application/json'.\n"
        "7. You must respond with the structured tool call or strictly valid JSON matching the schema."
    )


def build_user_prompt(context: EndpointContext) -> str:
    """Constructs the user prompt containing OpenAPI endpoint context."""
    return (
        f"Generate an APIPayload test request for the following endpoint:\n\n"
        f"Endpoint Path: {context.endpoint_path}\n"
        f"HTTP Method: {context.method}\n"
        f"Description: {context.description or 'No description provided.'}\n\n"
        f"Parameters:\n{json.dumps(context.required_params, indent=2)}\n\n"
        f"Request Body Schema:\n{json.dumps(context.request_schema, indent=2)}\n"
    )


async def generate_api_payload(
    context: EndpointContext,
    client: Optional[AsyncGroq] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> APIPayload:
    """Asynchronously generates a validated APIPayload using Groq and Llama 3.3 70B.

    Args:
        context: The OpenAPI endpoint context extracted from the specification.
        client: Optional pre-configured AsyncGroq client. If None, creates from GROQ_API_KEY.
        model: Model name to use (defaults to 'llama-3.3-70b-versatile').
        max_retries: Maximum number of retry attempts on rate limits or validation errors.
        initial_delay: Initial sleep duration in seconds before retry (doubles each attempt).

    Returns:
        APIPayload: Validated Pydantic model ready for execution.

    Raises:
        LLMValidationError: If the model repeatedly returns invalid JSON or fails Pydantic validation.
        LLMRateLimitError: If rate limits persist after all retry attempts.
        LLMClientError: For general unrecoverable API errors.
    """
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Allow fallback if client will be injected or mocked
            client = AsyncGroq(api_key="dummy_key_for_unauthenticated_init")
        else:
            client = AsyncGroq(api_key=api_key)

    target_model = model or DEFAULT_MODEL
    tools = [get_api_payload_tool_definition()]
    tool_choice = {"type": "function", "function": {"name": "submit_api_payload"}}

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(context)},
    ]

    current_delay = initial_delay
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Invoking Groq model %s for endpoint %s (attempt %d/%d)",
                target_model,
                context.endpoint_path,
                attempt,
                max_retries,
            )

            # Call Groq with tool calling & JSON object enforcement
            chat_completion = await client.chat.completions.create(
                model=target_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            choice = chat_completion.choices[0]
            message = choice.message

            # Extract JSON arguments from tool call or direct JSON message content
            raw_json_str: Optional[str] = None
            if message.tool_calls and len(message.tool_calls) > 0:
                raw_json_str = message.tool_calls[0].function.arguments
            elif message.content:
                raw_json_str = message.content

            if not raw_json_str:
                raise ValueError("Groq returned an empty response with no tool calls or content.")

            # Attempt to parse raw JSON
            try:
                parsed_dict = json.loads(raw_json_str)
            except json.JSONDecodeError as jde:
                raise ValueError(f"Malformed JSON returned by LLM: {jde}. Raw: {raw_json_str}") from jde

            # Validate against Pydantic APIPayload schema
            validated_payload = APIPayload.model_validate(parsed_dict)
            return validated_payload

        except RateLimitError as rle:
            last_error = rle
            logger.warning(
                "Groq RateLimitError encountered on attempt %d/%d: %s. Backing off for %.2f seconds.",
                attempt,
                max_retries,
                str(rle),
                current_delay,
            )
            if attempt == max_retries:
                raise LLMRateLimitError(
                    f"Groq RateLimitError after {max_retries} attempts: {str(rle)}"
                ) from rle
            await asyncio.sleep(current_delay)
            current_delay *= 2

        except (ValidationError, ValueError) as ve:
            last_error = ve
            logger.warning(
                "Validation/Parsing error on attempt %d/%d: %s",
                attempt,
                max_retries,
                str(ve),
            )
            if attempt == max_retries:
                raise LLMValidationError(
                    f"Failed to generate valid APIPayload after {max_retries} attempts. Error: {str(ve)}"
                ) from ve

            # Append corrective feedback to messages for self-correction in next attempt
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed validation with error: {str(ve)}.\n"
                        f"Please regenerate the request strictly complying with the APIPayload schema."
                    ),
                }
            )
            await asyncio.sleep(0.5)

        except APIError as apie:
            last_error = apie
            status_code = getattr(apie, "status_code", None)
            logger.error("Groq APIError (status %s): %s", status_code, str(apie))

            # Retry on transient server errors (500, 502, 503, 504)
            if status_code in (500, 502, 503, 504) and attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2
                continue

            raise LLMClientError(f"Groq API error encountered: {str(apie)}") from apie

        except Exception as exc:
            last_error = exc
            logger.error("Unexpected error during LLM payload generation: %s", str(exc))
            raise LLMClientError(f"Unexpected error in generate_api_payload: {str(exc)}") from exc

    raise LLMClientError(f"Exhausted all retries without generating valid payload. Last error: {last_error}")
