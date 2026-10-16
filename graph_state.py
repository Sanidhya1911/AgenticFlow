"""LangGraph multi-agent execution state schema.

This module defines the central state container passed between nodes in the LangGraph
execution graph (Planner, Executor, Corrector, Evaluator). The state is designed
as a TypedDict to conform natively to LangGraph's StateGraph interface, with an
equivalent Pydantic V2 validation model and utility factory.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field

from schemas import APIPayload, EndpointContext


class AgentState(TypedDict, total=False):
    """The central state dictionary passed between all nodes in the LangGraph multi-agent loop.

    Key Lifecycle & Roles:
    - Planner Agent: Takes `endpoint_context`, initializes `current_payload`, and sets `attempt_count = 0`.
    - Executor Agent: Dispatches `current_payload` over HTTP, updating `last_response_status`, `last_error_message`, and `attempt_count`.
    - Evaluator / Router: Analyzes status, toggling `success = True` if 2xx/expected, or routing to Corrector.
    - Corrector Agent: Analyzes `last_error_message` against `endpoint_context` to formulate an updated `current_payload`, incrementing `attempt_count`.
    """

    endpoint_context: Dict[str, Any]
    """The target OpenAPI endpoint specification details.

    Contains normalized operational metadata extracted by `openapi_parser.py`:
    - `endpoint_path`: Route pattern (e.g. '/orders/{orderId}')
    - `method`: HTTP verb ('GET', 'POST', 'PUT', etc.)
    - `description`: Operation summary and description
    - `required_params`: List of path, query, and header parameter definitions
    - `request_schema`: Fully dereferenced and flattened JSON schema for the request body

    Purpose:
    Serves as the foundational ground truth for both the Planner (when composing the initial test payload)
    and the Corrector (when cross-referencing server validation errors against schema constraints).
    """

    current_payload: Dict[str, Any]
    """The active HTTP request payload currently queued for dispatch or under correction.

    Matches the structure of the `APIPayload` schema:
    - `endpoint`: Evaluated concrete path with replaced parameters
    - `method`: HTTP method
    - `headers`: Dict of request headers
    - `json_body`: Body dictionary conforming to OpenAPI requirements
    - `query_params`: Dict of URL parameters

    Purpose:
    Passed directly to the Executor node to perform the HTTP request. Modified iteratively
    by the Corrector agent when fixing schema discrepancies or missing fields.
    """

    attempt_count: int
    """The cumulative number of execution/retry attempts completed for the current endpoint.

    Purpose:
    Provides cycle tracking to prevent infinite multi-agent feedback loops. LangGraph conditional
    edges inspect this value against `max_retries` (typically 3) to break recursion and terminate
    unrecoverable tests cleanly.
    """

    last_response_status: int
    """The numeric HTTP status code returned by the server on the most recent test attempt.

    Examples:
    - 200, 201: Success, signals test completion.
    - 400, 422: Schema/payload validation failure, signals Corrector intervention.
    - 401, 403: Authentication failure.
    - 404: Resource or path parameter error.
    - 500: Server-side failure.
    - 0: Network/connection timeout error before receiving an HTTP response.

    Purpose:
    Used by conditional routing edges and the Evaluator to assess whether execution succeeded or failed.
    """

    last_error_message: str
    """The body or diagnostic error message extracted from the failed HTTP response.

    Purpose:
    Supplies granular error traces (e.g., FastAPI/Pydantic validation details, SQL constraint messages,
    or raw response bodies) directly to the Corrector agent LLM prompt. Enables targeted self-correction
    without blind guessing.
    """

    success: bool
    """Boolean flag indicating whether the API test passed all validation criteria.

    Purpose:
    Controls termination routing in LangGraph. When `True`, the execution graph transitions to the END node.
    When `False` and `attempt_count < max_retries`, transitions to the Corrector node.
    """

    # Optional auxiliary telemetry fields
    max_retries: int
    """Maximum permitted retry attempts before marking the test step as permanently failed."""

    last_response_data: Optional[Union[Dict[str, Any], List[Any], str]]
    """The complete response payload returned from the most recent HTTP execution."""

    response_time_ms: Optional[float]
    """Round-trip latency in milliseconds for the most recent execution attempt."""

    history: List[Dict[str, Any]]
    """Audit log tracking payload iterations, status codes, and error diagnoses across all attempts."""


class AgentStateModel(BaseModel):
    """Pydantic V2 counterpart for strict validation and serialization of LangGraph state."""

    model_config = ConfigDict(extra="ignore")

    endpoint_context: Dict[str, Any] = Field(
        ...,
        description="Target endpoint details extracted from OpenAPI parser.",
    )
    current_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="The active JSON payload to be tested.",
    )
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Current retry attempt count.",
    )
    last_response_status: int = Field(
        default=0,
        description="HTTP status code from the last execution attempt.",
    )
    last_error_message: str = Field(
        default="",
        description="Error message or response body from the failed attempt.",
    )
    success: bool = Field(
        default=False,
        description="Flag indicating if the test passed.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum allowed retry attempts.",
    )
    last_response_data: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="Parsed response data from the last test call.",
    )
    response_time_ms: Optional[float] = Field(
        default=None,
        description="Response latency in milliseconds.",
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological log of execution attempts.",
    )

    def to_state_dict(self) -> AgentState:
        """Export as a clean TypedDict compatible with LangGraph."""
        return AgentState(
            endpoint_context=self.endpoint_context,
            current_payload=self.current_payload,
            attempt_count=self.attempt_count,
            last_response_status=self.last_response_status,
            last_error_message=self.last_error_message,
            success=self.success,
            max_retries=self.max_retries,
            last_response_data=self.last_response_data,
            response_time_ms=self.response_time_ms,
            history=self.history,
        )


def create_initial_state(
    endpoint_context: Union[Dict[str, Any], EndpointContext],
    initial_payload: Optional[Union[Dict[str, Any], APIPayload]] = None,
    max_retries: int = 3,
) -> AgentState:
    """Convenience factory to initialize a clean AgentState for a LangGraph workflow run.

    Args:
        endpoint_context: Extracted OpenAPI endpoint dictionary or EndpointContext model.
        initial_payload: Optional pre-generated APIPayload or dict. Defaults to empty dict.
        max_retries: Max retry attempts allowed for correction loops. Default is 3.

    Returns:
        AgentState: Normalized TypedDict ready to pass to `graph.invoke()`.
    """
    if isinstance(endpoint_context, EndpointContext):
        ctx_dict = endpoint_context.model_dump()
    elif isinstance(endpoint_context, dict):
        ctx_dict = endpoint_context
    else:
        raise TypeError(f"Expected EndpointContext or dict, got {type(endpoint_context)}")

    if initial_payload is None:
        payload_dict: Dict[str, Any] = {}
    elif isinstance(initial_payload, APIPayload):
        payload_dict = initial_payload.model_dump()
    elif isinstance(initial_payload, dict):
        payload_dict = initial_payload
    else:
        raise TypeError(f"Expected APIPayload or dict, got {type(initial_payload)}")

    return AgentState(
        endpoint_context=ctx_dict,
        current_payload=payload_dict,
        attempt_count=0,
        last_response_status=0,
        last_error_message="",
        success=False,
        max_retries=max_retries,
        last_response_data=None,
        response_time_ms=None,
        history=[],
    )
