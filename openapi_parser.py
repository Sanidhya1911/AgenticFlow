"""Root utility module for OpenAPI parser.

Exposes OpenAPIParser, parse_openapi_spec, extract_endpoints, and extract_endpoints_as_dict.
"""

from app.utils.openapi_parser import (
    OpenAPIParser,
    parse_openapi_spec,
    extract_endpoints,
    extract_endpoints_as_dict,
    SUPPORTED_HTTP_METHODS,
)

__all__ = [
    "OpenAPIParser",
    "parse_openapi_spec",
    "extract_endpoints",
    "extract_endpoints_as_dict",
    "SUPPORTED_HTTP_METHODS",
]
