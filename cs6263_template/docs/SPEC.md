# System Specification

> This is the source of truth for the project. The TA will feed this document
> to an LLM during grading and verify that the generated code passes the user
> story tests. Write it as if it must be enough for someone (or an LLM) to
> implement the project from scratch with no other reference.

## 1. Purpose and Scope

[One paragraph: what does the system do, who uses it, and what is explicitly
out of scope. Be specific. "An NLP question-answering system over a corpus of
NIST cybersecurity documents that returns a cited answer in under 3 seconds"
is good. "An AI assistant" is not.]

## 2. Component Inventory

Every component listed here must map to a source module under `src/myproject/`.
The grading script verifies this mapping via `grading/traceability.yaml`.

| Component | Source module | Responsibility |
|---|---|---|
| QueryRouter | src/myproject/router.py | Route incoming queries to the right pipeline |
| Retriever | src/myproject/retriever.py | Semantic search over the document corpus |
| Generator | src/myproject/generator.py | Generate cited answers via LLM |
| API | src/myproject/api.py | FastAPI HTTP interface |

## 3. Data Flow

[Reference the architecture diagram in docs/diagrams/architecture.png. Walk
through the data flow in prose: input format, transformations at each stage,
output format. Include error paths.]

## 4. Public Interfaces

This section is the contract. The user story tests import these exact module
paths and call these exact function signatures. The regenerated code must
match these signatures or the tests will fail.

### 4.1 HTTP API

```
POST /api/query
Content-Type: application/json

Request body:
{
  "text": "string, the user's question",
  "max_results": "integer, optional, default 5"
}

Response 200:
{
  "answer": "string, the generated answer with citations",
  "citations": [{"doc_id": "string", "snippet": "string"}],
  "latency_ms": "integer"
}

Response 400 (empty input):
{
  "error": "input text is required"
}
```

### 4.2 Python interfaces

```python
# src/myproject/router.py
def route_query(text: str, max_results: int = 5) -> dict: ...

# src/myproject/retriever.py
class Retriever:
    def __init__(self, index_path: str) -> None: ...
    def search(self, query: str, k: int = 5) -> list[dict]: ...

# src/myproject/generator.py
def generate_answer(query: str, contexts: list[dict]) -> dict: ...
```

## 5. External Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.11 | runtime |
| fastapi | 0.110.x | HTTP framework |
| anthropic | 0.40.x | LLM client |
| sentence-transformers | 2.7.x | embeddings |
| faiss-cpu | 1.8.x | vector search |
| pydantic | 2.x | request/response validation |

## 6. Configuration

The system reads configuration from environment variables. See `.env.example`
for the full list. Required keys: ANTHROPIC_API_KEY. Optional keys:
LOG_LEVEL (default INFO), MAX_CONTEXT_TOKENS (default 4000).

## 7. Model and Prompt Selection

[Justify your choice of model and prompting strategy. Why this model, why this
prompt structure, what alternatives you considered, and what known failure
modes you mitigate. This section directly informs the model card.]
