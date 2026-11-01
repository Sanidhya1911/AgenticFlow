"""LangGraph execution nodes for multi-agent REST API testing loop.

Provides the core async worker nodes for the LangGraph StateGraph:
- `executor_node`: Dispatches current HTTP request payload against the target
  OpenAPI endpoint via httpx.AsyncClient and processes HTTP status/diagnostics.
- `corrector_node`: Analyzes server error response bodies and schemas to generate
  corrected request payloads via the LLM client.
"""

import copy
import logging
import os
import sys
from typing import Any, Dict, Optional, Union

import httpx

from graph_state import AgentState
from schemas import APIPayload, EndpointContext
import llm_client
from llm_client import correct_payload as _llm_correct_payload

logger = logging.getLogger(__name__)

# Expose at module level for flexible patching and direct import
correct_payload = _llm_correct_payload


async def executor_node(state: AgentState) -> Dict[str, Any]:
    """Execute an HTTP request using current_payload against the target endpoint_context.

    Uses `httpx.AsyncClient` to dispatch the request.
    - If it receives a 2xx status, returns state updates with `success=True` and `last_response_status`.
    - If it receives a 4xx/5xx status (or non-2xx/network failure), captures the response body into
      `last_error_message`, updates `last_response_status`, increments `attempt_count`, and sets `success=False`.

    Args:
        state: The LangGraph AgentState containing endpoint_context, current_payload,
               and execution metrics.

    Returns:
        Dict[str, Any]: Dictionary representing the state updates for LangGraph.
    """
    # Extract endpoint_context and current_payload safely
    endpoint_context = state.get("endpoint_context", {})
    if hasattr(endpoint_context, "model_dump"):
        endpoint_context = endpoint_context.model_dump()

    current_payload = state.get("current_payload", {})
    if hasattr(current_payload, "model_dump"):
        current_payload = current_payload.model_dump()

    # Determine endpoint path, HTTP method, headers, params, and body
    endpoint = (
        current_payload.get("endpoint")
        or endpoint_context.get("endpoint_path")
        or endpoint_context.get("path")
        or "/"
    )
    method = (
        current_payload.get("method")
        or endpoint_context.get("method")
        or "GET"
    ).upper()

    headers = current_payload.get("headers") or {}
    query_params = (
        current_payload.get("query_params")
        or current_payload.get("params")
        or {}
    )
    json_body = current_payload.get("json_body")

    request_kwargs: Dict[str, Any] = {}
    if headers:
        request_kwargs["headers"] = headers
    if query_params:
        request_kwargs["params"] = query_params

    # Only send JSON body if provided and not an empty body on bodiless methods
    if json_body is not None:
        if method in ("GET", "DELETE", "HEAD", "OPTIONS") and json_body == {}:
            pass
        else:
            request_kwargs["json"] = json_body

    # Resolve target base URL
    base_url = (
        endpoint_context.get("base_url")
        or endpoint_context.get("server_url")
        or state.get("base_url")
        or os.getenv("API_BASE_URL", "http://localhost:8000")
    )
    if not endpoint_context.get("base_url") and isinstance(endpoint_context.get("servers"), list) and endpoint_context["servers"]:
        server_entry = endpoint_context["servers"][0]
        if isinstance(server_entry, dict) and "url" in server_entry:
            base_url = server_entry["url"]

    current_attempt = state.get("attempt_count", 0)

    try:
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.request(
                method=method,
                url=endpoint,
                **request_kwargs,
            )

        status_code = response.status_code

        if 200 <= status_code < 300:
            logger.info("Executor node received 2xx success (%d) for %s %s", status_code, method, endpoint)
            return {
                "success": True,
                "last_response_status": status_code,
            }
        else:
            error_body = response.text
            logger.warning("Executor node received error status (%d) for %s %s: %s", status_code, method, endpoint, error_body)
            return {
                "last_error_message": error_body,
                "last_response_status": status_code,
                "attempt_count": current_attempt + 1,
                "success": False,
            }

    except httpx.HTTPStatusError as hse:
        resp = hse.response
        logger.warning("HTTPStatusError raised (%d): %s", resp.status_code, resp.text)
        return {
            "last_error_message": resp.text,
            "last_response_status": resp.status_code,
            "attempt_count": current_attempt + 1,
            "success": False,
        }
    except httpx.RequestError as req_err:
        err_msg = str(req_err) or type(req_err).__name__
        logger.error("Network/connection RequestError for %s %s: %s", method, endpoint, err_msg)
        return {
            "last_error_message": err_msg,
            "last_response_status": 0,
            "attempt_count": current_attempt + 1,
            "success": False,
        }
    except Exception as exc:
        err_msg = str(exc) or type(exc).__name__
        logger.error("Unexpected error in executor_node for %s %s: %s", method, endpoint, err_msg)
        return {
            "last_error_message": err_msg,
            "last_response_status": 0,
            "attempt_count": current_attempt + 1,
            "success": False,
        }


async def corrector_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node that corrects current_payload using error feedback and LLM.

    Extracts `last_error_message` and `current_payload` from state and calls
    `llm_client.correct_payload(state)` to generate the corrected payload.

    Args:
        state: The LangGraph AgentState containing current_payload, last_error_message,
               and endpoint_context.

    Returns:
        Dict[str, Any]: Dictionary representing state updates with new `current_payload`.
    """
    last_error_message = state.get("last_error_message", "")
    current_payload = state.get("current_payload", {})

    logger.info(
        "Corrector node executing. Last error: '%s', Current payload: %s",
        last_error_message,
        current_payload,
    )

    # Dynamic function resolution ensuring mock patching works whether patched
    # on agent_nodes.correct_payload or llm_client.correct_payload
    fn = getattr(sys.modules[__name__], "correct_payload", None)
    if fn is not _llm_correct_payload and fn is not None:
        active_correct_payload = fn
    elif getattr(llm_client, "correct_payload", None) is not _llm_correct_payload:
        active_correct_payload = llm_client.correct_payload
    else:
        active_correct_payload = correct_payload

    new_payload = await active_correct_payload(state)

    if hasattr(new_payload, "model_dump"):
        new_payload = new_payload.model_dump()
    elif isinstance(new_payload, dict):
        new_payload = dict(new_payload)

    return {
        "current_payload": new_payload,
    }


__all__ = [
    "executor_node",
    "corrector_node",
    "correct_payload",
]
