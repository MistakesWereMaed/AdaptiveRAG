# Models

> Every model used by the project must be listed here with source, version,
> license, and download command. Model checkpoints must NOT be committed
> to the repo.

## Model 1: [name]

- **Source:** [HuggingFace org/model, Anthropic API, OpenAI API, etc.]
- **Identifier:** [e.g. claude-opus-4-5-20251101 or sentence-transformers/all-MiniLM-L6-v2]
- **Revision/version:** [git sha for HF, model id for API]
- **License:** [e.g. Apache 2.0, MIT, commercial API]
- **Size:** [e.g. 90 MB or "API only"]
- **Used for:** [e.g. query embedding, generation, classification]

### Download

```bash
make download-models
```

API-based models (Anthropic, OpenAI) do not require a download step; they
require credentials in `.env`.

For HuggingFace models:

```bash
mkdir -p models
huggingface-cli download <org/model> --revision <revision> \
    --local-dir models/<model_name>
```
