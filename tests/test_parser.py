"""Unit tests for OpenAPI parser and schema dereferencer."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from openapi_parser import (
    OpenAPIParser,
    parse_openapi_spec,
    extract_endpoints,
    extract_endpoints_as_dict,
)

SAMPLE_API_PATH = Path(__file__).parent.parent / "sample_api.json"


@pytest.fixture
def sample_spec_data():
    """Load sample OpenAPI 3.0 spec data."""
    assert SAMPLE_API_PATH.is_file(), f"sample_api.json not found at {SAMPLE_API_PATH}"
    with open(SAMPLE_API_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_load_json_file():
    """Test loading spec from a JSON file path."""
    parsed = parse_openapi_spec(SAMPLE_API_PATH)
    assert parsed.title == "AgenticFlow Sample E-Commerce API"
    assert parsed.version == "1.0.0"
    assert len(parsed.servers) > 0
    assert parsed.servers[0]["url"] == "https://api.agenticflow.io/v1"
    assert len(parsed.endpoints) == 7


def test_load_yaml_content():
    """Test loading spec from YAML content string."""
    yaml_content = """
    openapi: "3.0.0"
    info:
      title: "YAML Test API"
      version: "2.0.0"
    paths:
      /ping:
        get:
          summary: "Ping endpoint"
          responses:
            '200':
              description: "OK"
    """
    parser = OpenAPIParser.from_string(yaml_content, format_hint="yaml")
    spec = parser.parse_spec()
    assert spec.title == "YAML Test API"
    assert len(spec.endpoints) == 1
    assert spec.endpoints[0].path == "/ping"
    assert spec.endpoints[0].method == "GET"


def test_extract_all_endpoints(sample_spec_data):
    """Verify all 7 endpoint-method operations are extracted correctly."""
    endpoints = extract_endpoints(sample_spec_data)
    assert len(endpoints) == 7

    path_methods = {(ep.path, ep.method) for ep in endpoints}
    expected = {
        ("/orders", "GET"),
        ("/orders", "POST"),
        ("/orders/{orderId}", "GET"),
        ("/orders/{orderId}", "PUT"),
        ("/orders/{orderId}", "DELETE"),
        ("/categories", "GET"),
        ("/categories", "POST"),
    }
    assert path_methods == expected


def test_nested_ref_resolution(sample_spec_data):
    """Verify multi-level nested $ref references are completely resolved."""
    endpoints = extract_endpoints(sample_spec_data)
    create_order_op = next(ep for ep in endpoints if ep.path == "/orders" and ep.method == "POST")

    assert create_order_op.request_body is not None
    assert create_order_op.request_body.required is True
    assert create_order_op.request_body.content_type == "application/json"

    schema = create_order_op.request_body.schema_details

    # Level 1: CreateOrderRequest has properties: customer_id, items, shipping_address
    assert "properties" in schema
    props = schema["properties"]
    assert "customer_id" in props
    assert "items" in props
    assert "shipping_address" in props

    # Level 2: items is array of OrderItem -> resolved
    items_schema = props["items"]["items"]
    assert "properties" in items_schema
    item_props = items_schema["properties"]
    assert "product" in item_props
    assert "quantity" in item_props

    # Level 3: product is ProductItem -> resolved
    product_schema = item_props["product"]
    assert "properties" in product_schema
    assert "sku" in product_schema["properties"]
    assert "price" in product_schema["properties"]

    # Level 2 & 3: shipping_address -> coordinates (GeoLocation)
    addr_props = props["shipping_address"]["properties"]
    assert "coordinates" in addr_props
    coord_props = addr_props["coordinates"]["properties"]
    assert "latitude" in coord_props
    assert "longitude" in coord_props

    # Check that no unresolved $ref keys remain in the resolved schema tree
    def assert_no_unresolved_refs(obj):
        if isinstance(obj, dict):
            assert "$ref" not in obj, f"Found unresolved $ref in: {obj}"
            for v in obj.values():
                assert_no_unresolved_refs(v)
        elif isinstance(obj, list):
            for item in obj:
                assert_no_unresolved_refs(item)

    assert_no_unresolved_refs(schema)


def test_all_of_composition_flattening(sample_spec_data):
    """Verify allOf schema composition is cleanly merged and flattened."""
    endpoints = extract_endpoints(sample_spec_data)
    get_order_op = next(ep for ep in endpoints if ep.path == "/orders/{orderId}" and ep.method == "GET")

    response_200 = get_order_op.responses.get("200")
    assert response_200 is not None
    assert response_200.schema_details is not None

    schema = response_200.schema_details
    props = schema.get("properties", {})

    # Inherited from BaseEntity
    assert "id" in props
    assert "created_at" in props

    # Inherited from CreateOrderRequest
    assert "customer_id" in props
    assert "shipping_address" in props

    # Defined directly in OrderResponse
    assert "order_status" in props
    assert "total_amount" in props

    # Check required fields merged
    required = schema.get("required", [])
    assert "id" in required
    assert "customer_id" in required
    assert "order_status" in required


def test_parameter_extraction_and_ref_resolution(sample_spec_data):
    """Verify path, query, and ref parameters are extracted accurately."""
    endpoints = extract_endpoints(sample_spec_data)

    # /orders GET has query parameters (including resolved $ref parameters)
    list_orders_op = next(ep for ep in endpoints if ep.path == "/orders" and ep.method == "GET")
    param_map = {p.name: p for p in list_orders_op.parameters}

    assert "page" in param_map
    assert param_map["page"].in_location == "query"
    assert param_map["page"].schema_type == "integer"
    assert param_map["page"].required is False
    assert param_map["page"].schema_details.get("default") == 1

    assert "limit" in param_map
    assert param_map["limit"].in_location == "query"
    assert param_map["limit"].schema_type == "integer"

    assert "status" in param_map
    assert param_map["status"].schema_type == "string"

    # /orders/{orderId} GET has path parameter from path-level
    get_order_op = next(ep for ep in endpoints if ep.path == "/orders/{orderId}" and ep.method == "GET")
    assert len(get_order_op.parameters) >= 1
    order_id_param = next(p for p in get_order_op.parameters if p.name == "orderId")
    assert order_id_param.in_location == "path"
    assert order_id_param.required is True
    assert order_id_param.schema_type == "string"


def test_circular_ref_handling(sample_spec_data):
    """Verify recursive schemas (e.g. CategoryNode -> children -> CategoryNode) are handled without recursion error."""
    endpoints = extract_endpoints(sample_spec_data)
    post_cat_op = next(ep for ep in endpoints if ep.path == "/categories" and ep.method == "POST")

    assert post_cat_op.request_body is not None
    schema = post_cat_op.request_body.schema_details
    props = schema.get("properties", {})

    assert "id" in props
    assert "name" in props
    assert "children" in props

    # In children.items, circular ref should have been safely marked
    child_items = props["children"].get("items", {})
    assert child_items.get("$circularRef") == "#/components/schemas/CategoryNode"


def test_extract_endpoints_as_dict(sample_spec_data):
    """Verify to_dict export produces a JSON-serializable list of dictionaries."""
    dict_endpoints = extract_endpoints_as_dict(sample_spec_data)
    assert isinstance(dict_endpoints, list)
    assert len(dict_endpoints) == 7

    # Validate JSON serializability
    serialized = json.dumps(dict_endpoints)
    assert len(serialized) > 0
    deserialized = json.loads(serialized)
    assert isinstance(deserialized, list)
    assert deserialized[0]["path"] == "/orders"


def test_fastapi_endpoints(sample_spec_data):
    """Verify FastAPI endpoints /health and /api/v1/parse-spec."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["name"] == "AgenticFlow"

    # Parse spec endpoint
    res_parse = client.post("/api/v1/parse-spec", json={"spec": sample_spec_data})
    assert res_parse.status_code == 200
    data = res_parse.json()
    assert data["endpoint_count"] == 7
    assert data["spec_summary"]["title"] == "AgenticFlow Sample E-Commerce API"
    assert len(data["endpoints"]) == 7
