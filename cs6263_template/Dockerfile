# Multi-stage build. Final image runs as a non-root user.

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build
COPY requirements.txt ./
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.11.9-slim-bookworm

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash app
USER app
WORKDIR /home/app

# Install pinned wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --user --no-index --find-links=/wheels -r requirements.txt
ENV PATH=/home/app/.local/bin:$PATH

# Copy source
COPY --chown=app:app src ./src
COPY --chown=app:app pyproject.toml ./
RUN pip install --user -e .

# Application port
EXPOSE 8080

# Health endpoint must respond 200; docker-compose health check uses this
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

# Run application
CMD ["uvicorn", "myproject.api:app", "--host", "0.0.0.0", "--port", "8080"]
