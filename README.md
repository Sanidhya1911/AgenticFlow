# AgenticFlow 🚀

An autonomous, multi-agent workflow framework designed for dynamic REST API test suite generation and execution using **LangGraph**, **Groq API (Llama 3.3 70B)**, **FastAPI**, and **PostgreSQL**.

---

## Architecture Overview

```
                                      ┌──────────────────────┐
                                      │   OpenAPI 3.0 Spec   │
                                      │    (JSON / YAML)     │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │   openapi_parser.py   │
                                     │  (Recursive $ref &    │
                                     │   Flattening Engine)  │
                                     └───────────┬───────────┘
                                                 │
                                                 ▼
                    ┌────────────────────────────────────────────────────────┐
                    │            Normalized Endpoint Operations              │
                    │       (Path, Method, Params, Request Schemas)          │
                    └────────────────────────────┬───────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
        ┌─────────────────────────┐                             ┌─────────────────────────┐
        │   FastAPI Service       │                             │  LangGraph Multi-Agent  │
        │   (/api/v1/parse-spec)  │                             │  Execution Loop         │
        └─────────────────────────┘                             │  (Planner/Exec/Correct) │
                                                                └─────────────────────────┘
```

---

## Current Status: Phase 1 (OpenAPI 3.0 Parser & Setup)

### Completed Features:
- **FastAPI Core Setup**: Application bootstrap with CORS, health check (`GET /health`), and spec-parsing endpoints (`POST /api/v1/parse-spec`, `POST /api/v1/parse-file`).
- **LangGraph State Schema (`graph_state.py`)**: Defines [AgentState](file:///d:/AgenticFlow/graph_state.py) as a `TypedDict` for LangGraph `StateGraph` workflows, tracking `endpoint_context`, `current_payload`, `attempt_count`, `last_response_status`, `last_error_message`, and `success`. Includes Pydantic counterpart [AgentStateModel](file:///d:/AgenticFlow/graph_state.py) and [create_initial_state()](file:///d:/AgenticFlow/graph_state.py) factory.
- **Normalized Schema Models (`schemas.py` / `app/models/schema_models.py`)**: Strict Pydantic v2 data models ([APIPayload](file:///d:/AgenticFlow/schemas.py), [APIResponse](file:///d:/AgenticFlow/schemas.py), [EndpointContext](file:///d:/AgenticFlow/schemas.py)) with system-prompt descriptions for deterministic LLM tool-calling and agent handoffs.
- **Asynchronous Groq LLM Client (`llm_client.py`)**: Communicates with Llama 3.3 70B (`llama-3.3-70b-versatile`) via the official Groq SDK. Implements tool-calling (`submit_api_payload`) & JSON mode, exponential backoff on `RateLimitError`, auto-feedback loop on Pydantic `ValidationError`, and fatal `APIError` diagnostics.
- **Robust OpenAPI Parser (`openapi_parser.py` / `app/utils/openapi_parser.py`)**:
  - Ingestion of JSON and YAML OpenAPI 3.0.x / 3.1.x specifications.
  - Deep recursive `$ref` dereferencing across multi-level nested models.
  - Schema flattening & `allOf` composition merging into unified property dictionaries.
  - Circular reference detection & safe truncation (`$circularRef`) preventing recursion errors.
  - Extraction of path-level and operation-level parameters, query params, headers, and request bodies.
- **Sample OpenAPI 3.0 Spec (`sample_api.json`)**: Realistic e-commerce API testing nested models, address hierarchies, enums, array items, and recursive category nodes.
- **Test Suite (`tests/`)**: Comprehensive Pytest verification covering parser, schemas, and LLM client mocking with 21 passing tests.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v tests/test_parser.py
```

### 3. Usage Example (Python Script)
```python
from openapi_parser import extract_endpoints, extract_endpoints_as_dict

# Extract as normalized Pydantic/dataclass models
endpoints = extract_endpoints("sample_api.json")
for ep in endpoints:
    print(f"[{ep.method}] {ep.path} -> {ep.summary}")

# Extract as easy-to-traverse dictionaries for LLMs / LangGraph
dict_endpoints = extract_endpoints_as_dict("sample_api.json")
print(dict_endpoints[0])
```

### 4. Run FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
