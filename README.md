# AgenticFlow

[![CI Build](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square)](#)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

> Autonomous, self-healing REST API testing framework powered by OpenAPI specification parsing and stateful multi-agent execution loops.

---

## The Problem vs. The Solution

| Challenge | Legacy API Testing (Postman, Cypress) | AgenticFlow Architecture |
| :--- | :--- | :--- |
| **Payload Maintenance** | Static fixtures and hardcoded JSON mocks that silently break as schemas evolve. | Dynamic payload synthesis directly inferred from dereferenced OpenAPI 3.0/3.1 schemas. |
| **Error Handling** | Brittle assertion failures halt entire test suites; requires manual triage and code commits. | Autonomous self-healing loop: intercepts HTTP 4xx/5xx responses and repairs payloads in real time. |
| **Schema Drift** | High maintenance overhead; teams spend hours updating postman collections and test scripts. | Zero-script overhead; tests adapt automatically to spec updates, property renames, and enum additions. |
| **Edge-Case Coverage** | Limited to explicit test cases manually authored by developers. | LLM-driven constraint exploration targeting edge cases, boundary conditions, and nested dependencies. |

---

## Core Architecture

AgenticFlow is engineered around a deterministic, closed-loop multi-agent state machine built on **LangGraph**. The workflow decouples test generation, network dispatch, and payload remediation into isolated, auditable nodes.

```
                  ┌──────────────────────────────────────────────┐
                  │          OpenAPI 3.0 / 3.1 Spec Ingestion     │
                  │   (Recursive $ref Resolution & Flattening)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │                 Planner Node                 │
                  │  Synthesizes baseline request (Pydantic v2)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                               ┌──────────────────┐
                     ┌────────►│  Executor Node   │◄─────────┐
                     │         └─────────┬────────┘          │
                     │                   │                   │
                     │                   ▼                   │
                     │         ┌───────────────────┐         │
                     │         │ Conditional Router│         │
                     │         └───────┬───┬───────┘         │
                     │                 │   │                 │
    [attempt < max]  │   [HTTP 4xx/5xx]│   │[HTTP 2xx or     │
                     │                 │   │ max retries]    │
                     │                 ▼   ▼                 │
              ┌───────────────┐     ┌─────────┐              │
              │Corrector Node │     │   END   │              │
              │(Llama 3.3 70B)│     └─────────┘              │
              └───────┬───────┘                              │
                      │                                      │
                      └──────────────────────────────────────┘
```

### LangGraph Multi-Agent Execution Loop

1. **Planner Node**:
   - Parses the target endpoint operation metadata (`path`, `method`, `parameters`, `requestBody`).
   - Resolves required query parameters, path variables, and request body contracts into an initial `APIPayload` instance.
2. **Executor Node**:
   - Asynchronously dispatches the HTTP request using `httpx.AsyncClient`.
   - Records network diagnostics, latency, status code, and response payload into `AgentState`.
3. **Conditional Router**:
   - Evaluates execution telemetry:
     - **Success (`2xx`)**: Marks test as passed and routes to `END`.
     - **Terminal Failure (`attempt_count >= max_retries`)**: Terminates execution, persisting error logs for review.
     - **Remediable Failure (`4xx/5xx`)**: Increments attempt counter and routes to `Corrector`.
4. **Corrector Node**:
   - Consumes the raw API error payload, HTTP status code, failing payload, and OpenAPI schema definition.
   - Invokes Llama 3.3 70B via the Groq SDK with schema-bound tool definitions (`submit_api_payload`).
   - Generates a structurally and semantically corrected payload and cycles back into the `Executor Node`.

### Deterministic AI Handoffs via Pydantic v2

Non-deterministic LLM behavior is constrained through strict, runtime-enforced Pydantic schemas:

- **Strict Validation Contracts**: All LLM outputs pass through `APIPayload` and `AgentStateModel` schemas before dispatch. Schema discrepancies raise immediate validation exceptions rather than bubbling up as runtime HTTP bugs.
- **Automated Feedback Loop**: If an LLM response violates field types, enum values, or required constraints, the validation error is captured and reinjected into the prompt context for an immediate zero-shot retry.
- **State Immutability & Auditing**: The typed dictionary `AgentState` preserves full execution history across retry attempts, guaranteeing reproducibility across agent steps.

### State Flow Diagram

```mermaid
flowchart TD
    Start([Start Loop]) --> ExecutorNode[Executor Node\nhttpx Async Dispatch]

    ExecutorNode --> EvaluateStatus{HTTP Status Code?}

    EvaluateStatus -->|2xx Success| MarkSuccess[Set success = True\nCapture Telemetry]
    MarkSuccess --> EndSuccess([END: Test Passed])

    EvaluateStatus -->|4xx / 5xx Error| IncrementCounter[Increment attempt_count\nCapture last_error_message]
    IncrementCounter --> CheckRetries{attempt_count >= max_retries?}

    CheckRetries -->|Yes| EndFailure([END: Max Retries Exhausted])
    CheckRetries -->|No| CorrectorNode[Corrector Node\nGroq Llama 3.3 70B]

    CorrectorNode --> ValidatePayload{Pydantic v2\nValidation}
    ValidatePayload -->|Invalid| CorrectorNode
    ValidatePayload -->|Valid| UpdateState[Update current_payload\nin AgentState]
    UpdateState --> ExecutorNode
```

---

## Tech Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | **Python 3.11+** | Core runtime environment. |
| | **FastAPI** | High-performance asynchronous API layer and OpenAPI ingestion service. |
| | **LangGraph** | Cyclic state-machine orchestration for multi-agent loops. |
| | **Pydantic v2** | Strict schema validation, data serialization, and structured tool-calling. |
| | **HTTPX** | Fully asynchronous HTTP client for dynamic request execution. |
| **AI / Inference** | **Groq SDK** | Sub-second inference interface for LLM orchestration. |
| | **Llama 3.3 70B** | Primary LLM engine for reasoning, schema deduction, and error remediation. |
| **Data & Storage** | **PostgreSQL** | Relational datastore for test execution history, runs, and telemetry. |
| | **asyncpg** | High-throughput asynchronous PostgreSQL driver. |
| **Frontend** | **React 18** | Interactive dashboard for real-time test run observation and logs. |

---

## Getting Started

### Prerequisites

- **Python**: `3.11` or higher
- **Node.js**: `18.x` or higher (for the frontend dashboard)
- **PostgreSQL**: `14.x` or higher
- **Groq API Key**: [Obtain an API key from Groq Console](https://console.groq.com)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sanidhya1911/AgenticFlow.git
   cd AgenticFlow
   ```

2. **Initialize and activate a virtual environment**:
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure PostgreSQL**:
   ```sql
   -- Create database and user
   CREATE USER agenticflow WITH PASSWORD 'agenticflow_secret';
   CREATE DATABASE agenticflow_db OWNER agenticflow;
   GRANT ALL PRIVILEGES ON DATABASE agenticflow_db TO agenticflow;
   ```

5. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   # LLM Provider Configuration
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile

   # Database Configuration
   DATABASE_URL=postgresql+asyncpg://agenticflow:agenticflow_secret@localhost:5432/agenticflow_db

   # Application Configuration
   API_BASE_URL=http://localhost:8000
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   ```

---

## Running the Application

### 1. Launch FastAPI Backend

Start the API server with hot-reload enabled:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### 2. Launch React Frontend

In a separate terminal session, start the frontend client:

```bash
cd frontend
npm install
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173`.

### 3. Run Test Suite

Run the full automated test suite via pytest:

```bash
# Run all unit and integration tests
pytest -v

# Run parser tests
pytest -v tests/test_parser.py

# Run LangGraph state and execution node tests
pytest -v tests/test_graph_state.py tests/test_agent_nodes.py
```

---

## Programmatic Usage Example

```python
import asyncio
from openapi_parser import extract_endpoints
from graph_state import create_initial_state
from agent_nodes import executor_node, corrector_node
from langgraph.graph import StateGraph, START, END

async def main():
    # 1. Parse OpenAPI specification
    endpoints = extract_endpoints("sample_api.json")
    target_endpoint = endpoints[0]

    # 2. Initialize Agent State
    state = create_initial_state(
        endpoint_context=target_endpoint.model_dump(),
        max_retries=3
    )

    # 3. Assemble LangGraph Workflow
    workflow = StateGraph(state_schema=dict)
    workflow.add_node("executor", executor_node)
    workflow.add_node("corrector", corrector_node)

    def route_evaluation(current_state):
        if current_state.get("success") or current_state.get("attempt_count", 0) >= 3:
            return END
        return "corrector"

    workflow.add_edge(START, "executor")
    workflow.add_conditional_edges("executor", route_evaluation, ["corrector", END])
    workflow.add_edge("corrector", "executor")

    runner = workflow.compile()
    final_state = await runner.ainvoke(state)
    print(f"Workflow Finished. Success: {final_state.get('success')}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Contributing & License

Distributed under the MIT License. See `LICENSE` for more information.
Created and maintained by [Sanidhya Tulsyani](https://github.com/Sanidhya1911).
