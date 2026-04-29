# Makefile for the CS 6263 final project template.
#
# Every target here is referenced by the rubric. The TA runs these targets
# during grading. Do not rename them.
#
# Conventions:
#   * Targets that the TA runs are double-colon (.PHONY)
#   * All long-running targets emit reports under reports/

.PHONY: help test test-unit test-integration test-user-stories \
        lint format reproduce loadtest demo \
        download-data download-models \
        clean preflight regenerate

PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest

# Where source lives. The course pins this; do not change.
PACKAGE := myproject
SRC := src/$(PACKAGE)

# ---------------------------------------------------------------------------
help:
	@echo "Common targets:"
	@echo "  make test            run all test suites and write JUnit XML + coverage to reports/"
	@echo "  make lint            run ruff + black --check + mypy"
	@echo "  make format          apply black formatting in place"
	@echo "  make reproduce       full clean replay: download data + models, run pipeline, run tests"
	@echo "  make loadtest        run the load test against the live app"
	@echo "  make demo            exercise every user story against the live app"
	@echo "  make download-data   fetch datasets listed in docs/DATA.md"
	@echo "  make download-models fetch model checkpoints listed in docs/MODELS.md"
	@echo "  make regenerate      regenerate source from spec via LLM (rubric Spec test)"
	@echo "  make preflight       run every automated grading check locally"
	@echo "  make clean           remove generated artifacts"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: reports
	$(PYTEST) tests/unit \
	  --junitxml=reports/unit.xml
	$(PYTEST) tests/integration \
	  --junitxml=reports/integration.xml
	$(PYTEST) tests/user_stories \
	  --junitxml=reports/user_stories.xml
	$(PYTEST) tests/edge \
	  --junitxml=reports/edge.xml || true
	$(PYTEST) --cov=$(PACKAGE) --cov-report=xml:reports/coverage.xml \
	          --cov-report=html:reports/coverage_html \
	          --cov-fail-under=70 \
	          tests/unit tests/integration

test-unit: reports
	$(PYTEST) tests/unit --junitxml=reports/unit.xml

test-integration: reports
	$(PYTEST) tests/integration --junitxml=reports/integration.xml

test-user-stories: reports
	$(PYTEST) tests/user_stories --junitxml=reports/user_stories.xml

# ---------------------------------------------------------------------------
# Lint / format / static checks
# ---------------------------------------------------------------------------
lint:
	ruff check $(SRC) tests/
	black --check $(SRC) tests/
	mypy $(SRC)

format:
	ruff check --fix $(SRC) tests/
	black $(SRC) tests/

# ---------------------------------------------------------------------------
# Reproduce: full clean replay
# ---------------------------------------------------------------------------
reproduce: reports
	@echo "[reproduce] full clean replay; budget 30 min on documented hardware"
	$(MAKE) download-data
	$(MAKE) download-models
	$(PYTHON) -m $(PACKAGE).pipeline --sample
	$(MAKE) test
	@echo "[reproduce] done. Compare reports/ values to the README Results table."

# ---------------------------------------------------------------------------
# Data and model download (skeletons — implement per project)
# ---------------------------------------------------------------------------
download-data:
	@echo "[download-data] implement per docs/DATA.md"
	# Example:
	# mkdir -p data/raw
	# curl -L -o data/raw/dataset.tar.gz https://example.com/dataset.tar.gz
	# echo "<expected sha256>  data/raw/dataset.tar.gz" | sha256sum -c -

download-models:
	@echo "[download-models] implement per docs/MODELS.md"
	# Example:
	# mkdir -p models
	# huggingface-cli download org/model --local-dir models/model

# ---------------------------------------------------------------------------
# Stress and load
# ---------------------------------------------------------------------------
loadtest: reports
	@echo "[loadtest] requires the app to be running (docker compose up)"
	locust -f tests/load/locustfile.py \
	  --headless -u 20 -r 5 -t 60s \
	  --host=$${APP_URL:-http://localhost:8080} \
	  --csv=reports/loadtest \
	  || true
	$(PYTHON) -c "import json,csv,pathlib; \
	rows=list(csv.DictReader(open('reports/loadtest_stats.csv'))); \
	pathlib.Path('reports/benchmarks.json').write_text(json.dumps(rows, indent=2))"

# ---------------------------------------------------------------------------
# Demo and regeneration
# ---------------------------------------------------------------------------
demo:
	bash scripts/demo.sh

regenerate:
	bash scripts/regenerate.sh

preflight:
	bash scripts/preflight.sh

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
reports:
	@mkdir -p reports

clean:
	rm -rf reports regenerated .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
