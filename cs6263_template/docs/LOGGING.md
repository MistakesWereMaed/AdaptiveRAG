# Logging

> The Logging category test (5 points) is graded by the TA tracing one
> request end-to-end through `docker compose logs -f app` using its request
> ID. This document includes a worked example so the TA knows what to look for.

## Format

All log entries are JSON, one entry per line, written to stdout. Format:

```json
{
  "timestamp": "2026-04-28T14:32:01.234Z",
  "level": "INFO",
  "module": "myproject.api",
  "request_id": "req_a1b2c3d4",
  "message": "received query",
  "extra": {"text_length": 42}
}
```

Required fields: `timestamp`, `level`, `module`, `request_id`, `message`.

## Levels

- `DEBUG` — internal trace, off by default
- `INFO` — normal events (request received, response sent)
- `WARNING` — recoverable issues (slow upstream, retry triggered)
- `ERROR` — request failed despite retries

The application sets level via `LOG_LEVEL` env var; default is `INFO`.

## Request ID Propagation

Every incoming HTTP request is assigned a `request_id` by the API middleware.
The ID is then included in the logging context for that request, so every
component touched by the request emits log lines tagged with the same ID.

## Worked Example

Below is a full request lifecycle visible from logs alone, captured with:

```bash
docker compose logs -f app | grep req_a1b2c3d4
```

```json
{"timestamp":"2026-04-28T14:32:01.234Z","level":"INFO","module":"myproject.api","request_id":"req_a1b2c3d4","message":"received query","extra":{"text_length":18}}
{"timestamp":"2026-04-28T14:32:01.241Z","level":"INFO","module":"myproject.router","request_id":"req_a1b2c3d4","message":"routing to retrieval pipeline","extra":{"pipeline":"qa"}}
{"timestamp":"2026-04-28T14:32:01.298Z","level":"INFO","module":"myproject.retriever","request_id":"req_a1b2c3d4","message":"index search complete","extra":{"k":5,"hits":5,"ms":52}}
{"timestamp":"2026-04-28T14:32:01.302Z","level":"INFO","module":"myproject.generator","request_id":"req_a1b2c3d4","message":"calling LLM","extra":{"model":"claude-opus-4-5-20251101","context_tokens":1840}}
{"timestamp":"2026-04-28T14:32:03.611Z","level":"INFO","module":"myproject.generator","request_id":"req_a1b2c3d4","message":"LLM response received","extra":{"output_tokens":312,"ms":2308}}
{"timestamp":"2026-04-28T14:32:03.620Z","level":"INFO","module":"myproject.api","request_id":"req_a1b2c3d4","message":"response sent","extra":{"status":200,"total_ms":2386}}
```

The TA can read this trace top to bottom to confirm:
- request arrival,
- pipeline routing,
- retrieval,
- LLM call,
- response delivery.

If the TA can do this for any request without reading source code, the Logging
category scores full credit.
