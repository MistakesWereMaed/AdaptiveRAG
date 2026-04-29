# Model Card

> Required for any system that calls an LLM. The grading script checks for the
> four sections below by name. Each section needs substantive content; a one-line
> placeholder will not count.

## Intended Use

[Describe who should use the system, in what context, for what purpose. Be
specific. "Researchers querying a public NIST cybersecurity corpus to find
relevant standards" is good. "AI assistant" is not.]

## Limitations

[Document what the system does poorly or cannot do. Examples to consider:
the model's knowledge cutoff date; languages other than English; queries
outside the indexed corpus; questions requiring real-time data; long
multi-hop reasoning; numerical computation; etc.]

## Risks

[Document risks and mitigations. Required topics:
- Hallucination: how the system mitigates fabricated citations
- Prompt injection: how user input is bounded before being passed to the LLM
- Bias: known biases in the model or corpus that could skew outputs
- Privacy: what user input is logged and for how long
- Cost: rate limiting and per-request budget controls]

## Out of Scope

[Document explicit non-goals. Examples: medical advice, legal advice,
real-time conversation, code execution, file uploads, multimedia, etc.
The system should refuse or redirect requests in these areas.]
