"""FastAPI application entrypoint for AgenticFlow."""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.models.schema_models import EndpointOperation, ParsedApiSpec
from app.utils.openapi_parser import OpenAPIParser, parse_openapi_spec

app = FastAPI(
    title="AgenticFlow API",
    version="0.1.0",
    description="Agentic Workflow Framework for Automated REST API Testing using LangGraph & Groq Llama 3.3 70B",
)

# Enable CORS for future React dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ParseSpecRequest(BaseModel):
    """Request model for inline spec parsing."""
    spec: Dict[str, Any] = Field(description="Raw OpenAPI 3.0 specification dictionary")


class ParseSpecResponse(BaseModel):
    """Response model for parsed spec."""
    spec_summary: Dict[str, Any]
    endpoint_count: int
    endpoints: List[EndpointOperation]


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "AgenticFlow API", "version": "0.1.0"}


@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """Root endpoint with service metadata."""
    return {
        "name": "AgenticFlow",
        "description": "Multi-agent framework for OpenAPI test generation and execution",
        "documentation": "/docs",
        "health": "/health",
    }


@app.post(
    "/api/v1/parse-spec",
    response_model=ParseSpecResponse,
    tags=["OpenAPI Parser"],
    summary="Parse raw OpenAPI 3.0 specification JSON",
)
async def parse_spec_endpoint(request: ParseSpecRequest) -> ParseSpecResponse:
    """Accepts an in-memory OpenAPI specification dictionary and returns normalized endpoints."""
    try:
        parser = OpenAPIParser(request.spec)
        parsed = parser.parse_spec()
        return ParseSpecResponse(
            spec_summary={
                "title": parsed.title,
                "version": parsed.version,
                "description": parsed.description,
                "servers": parsed.servers,
            },
            endpoint_count=len(parsed.endpoints),
            endpoints=parsed.endpoints,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse OpenAPI specification: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/parse-file",
    response_model=ParseSpecResponse,
    tags=["OpenAPI Parser"],
    summary="Upload and parse an OpenAPI specification file (.json or .yaml)",
)
async def parse_file_endpoint(file: UploadFile = File(...)) -> ParseSpecResponse:
    """Accepts an uploaded JSON or YAML OpenAPI specification file and returns normalized endpoints."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    content_bytes = await file.read()
    content_str = content_bytes.decode("utf-8")

    format_hint = "yaml" if file.filename.endswith((".yaml", ".yml")) else "json"

    try:
        parser = OpenAPIParser.from_string(content_str, format_hint=format_hint)
        parsed = parser.parse_spec()
        return ParseSpecResponse(
            spec_summary={
                "title": parsed.title,
                "version": parsed.version,
                "description": parsed.description,
                "servers": parsed.servers,
            },
            endpoint_count=len(parsed.endpoints),
            endpoints=parsed.endpoints,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse uploaded specification: {str(exc)}",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
