# Datasets

> Every dataset used by the project must be listed here with source, version,
> license, and download command. Datasets must NOT be committed to the repo.

## Dataset 1: [name]

- **Source URL:** https://example.com/dataset
- **Version:** YYYY-MM-DD or release tag
- **sha256:** [computed hash of the downloaded archive]
- **License:** [e.g. CC BY 4.0, public domain, custom]
- **Size:** [e.g. 2.4 GB compressed]
- **Cite as:** [recommended citation if any]

### Download

```bash
make download-data
```

Or manually:

```bash
mkdir -p data/raw
curl -L -o data/raw/dataset.tar.gz https://example.com/dataset.tar.gz
echo "<expected sha256>  data/raw/dataset.tar.gz" | sha256sum -c -
tar -xzf data/raw/dataset.tar.gz -C data/
```

### Preprocessing

[Document any preprocessing applied before training/indexing. The
preprocessing must be deterministic given the seed in grading/manifest.yaml.]
