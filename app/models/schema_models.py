"""Normalized data models for OpenAPI endpoints and request/response schemas.

These models provide structured, deterministic objects for consumption by LangGraph
agents (Planner, Executor, Corrector) and LLM tool calling via Groq.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """Represents an extracted path, query, header, or cookie parameter."""

    name: str
    in_location: str = Field(description="Parameter location: path, query, header, or cookie")
    required: bool = False
    description: Optional[str] = None
    schema_type: Optional[str] = None
    schema_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dereferenced and normalized JSON schema for this parameter",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RequestBodySchema(BaseModel):
    """Represents a dereferenced and normalized request body."""

    content_type: str = "application/json"
    required: bool = False
    description: Optional[str] = None
    schema_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flattened, dereferenced JSON schema ready for LLM payload generation",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ResponseSchema(BaseModel):
    """Represents an endpoint response definition."""

    status_code: str
    description: str = ""
    content_type: Optional[str] = None
    schema_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EndpointOperation(BaseModel):
    """Normalized representation of a single API operation (endpoint + HTTP method)."""

    path: str
    method: str
    operation_id: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    parameters: List[ParameterSchema] = Field(default_factory=list)
    request_body: Optional[RequestBodySchema] = None
    responses: Dict[str, ResponseSchema] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean, JSON-serializable dictionary."""
        return self.model_dump()


class ParsedApiSpec(BaseModel):
    """Overall parsed API spec container."""

    title: str
    version: str
    description: Optional[str] = None
    servers: List[Dict[str, Any]] = Field(default_factory=list)
    endpoints: List[EndpointOperation] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
