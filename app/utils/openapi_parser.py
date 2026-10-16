"""OpenAPI 3.0 Parser and Schema Dereferencer.

Reads standard OpenAPI 3.0 JSON/YAML files or dictionaries, resolves and flattens
nested and circular `$ref` schema references, and extracts normalized endpoints,
HTTP methods, parameters, and request/response schemas for downstream LLM agents.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

from app.models.schema_models import (
    EndpointOperation,
    ParameterSchema,
    ParsedApiSpec,
    RequestBodySchema,
    ResponseSchema,
)

# Standard HTTP methods supported in OpenAPI
SUPPORTED_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}


class OpenAPIParser:
    """Robust parser for OpenAPI 3.0 JSON and YAML specifications.

    Resolves nested and circular `$ref` pointers and normalizes endpoint schemas
    for seamless integration with LLMs and multi-agent workflows.
    """

    def __init__(self, raw_spec: Dict[str, Any]):
        """Initialize parser with raw OpenAPI dictionary."""
        self.raw_spec = raw_spec
        self.doc = copy.deepcopy(raw_spec)
        self._ref_cache: Dict[str, Any] = {}

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "OpenAPIParser":
        """Load and parse an OpenAPI specification from a JSON or YAML file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenAPI spec file not found: {path}")

        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        elif path.suffix.lower() == ".json":
            data = json.loads(content)
        else:
            # Attempt JSON first, then YAML fallback
            try:
                data = json.loads(content)
            except Exception:
                data = yaml.safe_load(content)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid OpenAPI spec format: expected a dictionary at root in {path}")

        return cls(data)

    @classmethod
    def from_string(cls, content: str, format_hint: str = "json") -> "OpenAPIParser":
        """Load and parse an OpenAPI specification from raw JSON or YAML text."""
        if format_hint.lower() == "json":
            try:
                data = json.loads(content)
            except Exception:
                data = yaml.safe_load(content)
        else:
            try:
                data = yaml.safe_load(content)
            except Exception:
                data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("Invalid OpenAPI spec content: expected a dictionary at root")

        return cls(data)

    def resolve_json_pointer(self, ref: str) -> Any:
        """Resolve a local JSON Pointer (e.g. `#/components/schemas/OrderItem`).

        Supports RFC 6901 escaping (~1 for '/', ~0 for '~').
        """
        if not ref.startswith("#/"):
            raise ValueError(f"External $ref pointers not supported yet: {ref}")

        parts = ref[2:].split("/")
        current = self.doc
        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                raise KeyError(f"Failed to resolve pointer '{ref}': key '{part}' not found")
        return current

    def resolve_schema(
        self,
        schema: Any,
        ancestor_refs: Optional[Set[str]] = None,
    ) -> Any:
        """Recursively resolves all `$ref` references and flattens schemas.

        Guards against cyclic references (e.g. self-referencing tree structures)
        and merges `allOf` schemas into a single unified object representation.
        """
        if ancestor_refs is None:
            ancestor_refs = set()

        if not isinstance(schema, dict):
            if isinstance(schema, list):
                return [self.resolve_schema(item, ancestor_refs) for item in schema]
            return schema

        # Handle $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in ancestor_refs:
                # Circular reference detected! Break loop cleanly with a marker
                ref_name = ref.split("/")[-1]
                return {
                    "type": "object",
                    "title": ref_name,
                    "$circularRef": ref,
                    "description": f"Recursive/circular reference to {ref}",
                }

            # Resolve reference
            target = self.resolve_json_pointer(ref)
            target_copy = copy.deepcopy(target)

            new_ancestors = ancestor_refs | {ref}
            resolved_target = self.resolve_schema(target_copy, new_ancestors)

            # If the current schema had other fields alongside $ref, merge them
            merged = {}
            if isinstance(resolved_target, dict):
                merged.update(resolved_target)
            for k, v in schema.items():
                if k != "$ref":
                    merged[k] = self.resolve_schema(v, ancestor_refs)
            return merged

        # Handle allOf composition by merging properties and required arrays
        if "allOf" in schema and isinstance(schema["allOf"], list):
            merged_schema: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            # Preserve top-level fields (e.g., description, title)
            for k, v in schema.items():
                if k != "allOf":
                    merged_schema[k] = self.resolve_schema(v, ancestor_refs)

            for sub_schema in schema["allOf"]:
                resolved_sub = self.resolve_schema(sub_schema, ancestor_refs)
                if isinstance(resolved_sub, dict):
                    if "properties" in resolved_sub:
                        merged_schema["properties"].update(resolved_sub["properties"])
                    if "required" in resolved_sub and isinstance(resolved_sub["required"], list):
                        merged_schema["required"].extend(resolved_sub["required"])
                    for key, val in resolved_sub.items():
                        if key not in ("properties", "required", "type") and key not in merged_schema:
                            merged_schema[key] = val

            # Deduplicate required list while preserving order
            merged_schema["required"] = list(dict.fromkeys(merged_schema["required"]))
            if not merged_schema["required"]:
                merged_schema.pop("required", None)
            return merged_schema

        # Process anyOf and oneOf
        result: Dict[str, Any] = {}
        for key, val in schema.items():
            if key in ("anyOf", "oneOf") and isinstance(val, list):
                result[key] = [self.resolve_schema(item, ancestor_refs) for item in val]
            elif key == "properties" and isinstance(val, dict):
                result["properties"] = {
                    prop_name: self.resolve_schema(prop_val, ancestor_refs)
                    for prop_name, prop_val in val.items()
                }
            elif key == "items":
                result["items"] = self.resolve_schema(val, ancestor_refs)
            elif isinstance(val, dict):
                result[key] = self.resolve_schema(val, ancestor_refs)
            elif isinstance(val, list):
                result[key] = [self.resolve_schema(item, ancestor_refs) for item in val]
            else:
                result[key] = val

        return result

    def _extract_parameter(self, param_raw: Dict[str, Any]) -> ParameterSchema:
        """Extract and normalize a parameter, dereferencing if it uses $ref."""
        if "$ref" in param_raw:
            param_raw = self.resolve_schema(param_raw)

        name = param_raw.get("name", "")
        in_loc = param_raw.get("in", "query")
        required = bool(param_raw.get("required", in_loc == "path"))
        description = param_raw.get("description")

        param_schema = param_raw.get("schema", {})
        resolved_schema = self.resolve_schema(param_schema)

        schema_type = resolved_schema.get("type") if isinstance(resolved_schema, dict) else None

        return ParameterSchema(
            name=name,
            in_location=in_loc,
            required=required,
            description=description,
            schema_type=schema_type,
            schema_details=resolved_schema if isinstance(resolved_schema, dict) else {},
        )

    def _extract_request_body(self, body_raw: Dict[str, Any]) -> Optional[RequestBodySchema]:
        """Extract and dereference a request body specification."""
        if not body_raw:
            return None

        if "$ref" in body_raw:
            body_raw = self.resolve_schema(body_raw)

        required = bool(body_raw.get("required", False))
        description = body_raw.get("description")
        content = body_raw.get("content", {})

        # Primary content type preference: application/json -> first available
        target_content_type = "application/json"
        body_content = content.get(target_content_type)

        if not body_content and content:
            target_content_type = next(iter(content.keys()))
            body_content = content[target_content_type]

        if not body_content:
            return None

        raw_schema = body_content.get("schema", {})
        resolved_schema = self.resolve_schema(raw_schema)

        return RequestBodySchema(
            content_type=target_content_type,
            required=required,
            description=description,
            schema_details=resolved_schema if isinstance(resolved_schema, dict) else {},
        )

    def _extract_responses(self, responses_raw: Dict[str, Any]) -> Dict[str, ResponseSchema]:
        """Extract and dereference response definitions."""
        extracted: Dict[str, ResponseSchema] = {}
        for status_code, resp_raw in responses_raw.items():
            if "$ref" in resp_raw:
                resp_raw = self.resolve_schema(resp_raw)

            desc = resp_raw.get("description", "")
            content = resp_raw.get("content", {})
            content_type = None
            schema_details = None

            if content:
                content_type = "application/json" if "application/json" in content else next(iter(content.keys()))
                raw_schema = content[content_type].get("schema", {})
                schema_details = self.resolve_schema(raw_schema)

            extracted[str(status_code)] = ResponseSchema(
                status_code=str(status_code),
                description=desc,
                content_type=content_type,
                schema_details=schema_details if isinstance(schema_details, dict) else None,
            )
        return extracted

    def parse_spec(self) -> ParsedApiSpec:
        """Parse complete specification into a structured ParsedApiSpec instance."""
        info = self.doc.get("info", {})
        title = info.get("title", "Untitled API")
        version = info.get("version", "1.0.0")
        description = info.get("description")
        servers = self.doc.get("servers", [])

        endpoints = self.extract_endpoints()

        return ParsedApiSpec(
            title=title,
            version=version,
            description=description,
            servers=servers,
            endpoints=endpoints,
        )

    def extract_endpoints(self) -> List[EndpointOperation]:
        """Extract all endpoints, methods, parameters, and dereferenced request schemas."""
        endpoints: List[EndpointOperation] = []
        paths = self.doc.get("paths", {})

        for path_str, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            # Path-level parameters apply to all operations under this path
            path_level_params_raw = path_item.get("parameters", [])
            path_level_params = [
                self._extract_parameter(p) for p in path_level_params_raw if isinstance(p, dict)
            ]

            for method_key, op_raw in path_item.items():
                method_lower = method_key.lower()
                if method_lower not in SUPPORTED_HTTP_METHODS:
                    continue

                if not isinstance(op_raw, dict):
                    continue

                operation_id = op_raw.get("operationId")
                summary = op_raw.get("summary")
                description = op_raw.get("description")
                tags = op_raw.get("tags", [])

                # Operation-level parameters (overriding or supplementing path params)
                op_params_raw = op_raw.get("parameters", [])
                op_params = [
                    self._extract_parameter(p) for p in op_params_raw if isinstance(p, dict)
                ]

                # Combine path and operation parameters (operation param overrides path param by (name, in))
                param_dict = {(p.name, p.in_location): p for p in path_level_params}
                for p in op_params:
                    param_dict[(p.name, p.in_location)] = p
                combined_params = list(param_dict.values())

                # Request body
                request_body = self._extract_request_body(op_raw.get("requestBody", {}))

                # Responses
                responses = self._extract_responses(op_raw.get("responses", {}))

                endpoint = EndpointOperation(
                    path=path_str,
                    method=method_lower.upper(),
                    operation_id=operation_id,
                    summary=summary,
                    description=description,
                    tags=tags,
                    parameters=combined_params,
                    request_body=request_body,
                    responses=responses,
                )
                endpoints.append(endpoint)

        return endpoints

    def extract_endpoints_as_dict(self) -> List[Dict[str, Any]]:
        """Extract endpoints as a normalized, easy-to-traverse list of Python dictionaries."""
        return [ep.to_dict() for ep in self.extract_endpoints()]


# Convenience Functional Interface
def parse_openapi_spec(source: Union[str, Path, Dict[str, Any]]) -> ParsedApiSpec:
    """Convenience function to parse an OpenAPI spec from file path or dictionary."""
    if isinstance(source, (str, Path)):
        parser = OpenAPIParser.from_file(source)
    elif isinstance(source, dict):
        parser = OpenAPIParser(source)
    else:
        raise TypeError(f"Expected file path or dict, got {type(source)}")
    return parser.parse_spec()


def extract_endpoints(source: Union[str, Path, Dict[str, Any]]) -> List[EndpointOperation]:
    """Convenience function to extract normalized EndpointOperation dataclasses."""
    if isinstance(source, (str, Path)):
        parser = OpenAPIParser.from_file(source)
    elif isinstance(source, dict):
        parser = OpenAPIParser(source)
    else:
        raise TypeError(f"Expected file path or dict, got {type(source)}")
    return parser.extract_endpoints()


def extract_endpoints_as_dict(source: Union[str, Path, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function to extract normalized Python dictionaries for LLMs."""
    if isinstance(source, (str, Path)):
        parser = OpenAPIParser.from_file(source)
    elif isinstance(source, dict):
        parser = OpenAPIParser(source)
    else:
        raise TypeError(f"Expected file path or dict, got {type(source)}")
    return parser.extract_endpoints_as_dict()
