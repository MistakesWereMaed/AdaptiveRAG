# CS 6263 Final Project Template

This is the starting point for your CS 6263 NLP and Agentic AI final project.
Clone it, push it to your own public GitHub repository, build your project
inside it, and submit the GitHub URL by **May 10, 11:59 PM**.

The grading rubric is in `nlp_final_project_rubric_v10.docx` (distributed
alongside this template).

## What is in here

| Path | Purpose |
|---|---|
| `README.md` | Your project's user-facing README. Replace placeholders. |
| `Makefile` | Command surface the TA uses during grading. Do not rename targets. |
| `Dockerfile`, `docker-compose.yml` | Reference container setup. Modify as needed. |
| `pyproject.toml` | Pinned dependencies + lint/test/coverage config. |
| `requirements.txt` | Mirror of pyproject deps for `pip-audit`. |
| `.env.example` | Template for environment variables. Copy to `.env` locally. |
| `.gitignore` | Excludes `.env`, generated artifacts, data, models. |
| `CONTRIBUTIONS.md` | Team roster with role and percentage contribution. |
| `docs/SPEC.md` | The spec. The source of truth for grading. Take this seriously. |
| `docs/STORIES.md` | User stories with manual walkthrough steps. |
| `docs/MODEL_CARD.md` | Model intended use, limitations, risks, out of scope. |
| `docs/REPRODUCE.md` | Reproducibility procedure and expected numbers. |
| `docs/DATA.md`, `docs/MODELS.md` | Datasets and model checkpoints. |
| `docs/benchmarks.md` | Performance numbers from your own load test. |
| `docs/LOGGING.md` | Log format and worked request-trace example. |
| `docs/usage.md` | Per-feature usage documentation. |
| `grading/manifest.yaml` | Pinned environment + commit_sha for grading. |
| `grading/traceability.yaml` | Story → spec → modules → tests mapping. |
| `grading/grade.py` | Course-issued grader script. Do not modify. |
| `scripts/regenerate_prompt.md` | Course-issued spec regeneration prompt. **Do not modify.** |
| `scripts/regenerate.sh` | Spec regeneration runner. |
| `scripts/preflight.sh` | Local TA dry-run. **Run this before pushing.** |
| `scripts/demo.sh` | End-to-end demo against the running app. |
| `src/myproject/` | Your application code. The package name `myproject` is pinned. |
| `tests/unit/`, `tests/integration/` | Pytest tests for source modules. |
| `tests/user_stories/` | One test per user story. |
| `tests/edge/` | Linguistic edge case tests. |
| `tests/load/locustfile.py` | Load test driver. |
| `reports/walkthrough.md` | Form the TA fills in during Phase 2 and Phase 3. |

## Quick start for new teams

```bash
# 1. Clone this template, then push to your own public GitHub repo
git clone <this-template-url> myteam-final-project
cd myteam-final-project
git remote set-url origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main

# 2. Install dev dependencies locally
pip install -e ".[dev]"

# 3. Replace every "[TO BE FILLED]" and "[your-...]" placeholder
#    Start with: README.md, docs/SPEC.md, docs/STORIES.md, CONTRIBUTIONS.md

# 4. Implement your project under src/myproject/

# 5. Verify locally before pushing
bash scripts/preflight.sh

# 6. When preflight passes, commit, push, and submit your GitHub URL
```

## What you must change

Every file above contains placeholders. Search for these strings and replace
them with content specific to your project:

- `[TO BE FILLED]`
- `[your-...]`
- `[Project Title — replace ...]`
- `[Name 1]`, `[Name 2]`, ...
- `[name]` in docs/DATA.md and docs/MODELS.md

## What you must NOT change

- `scripts/regenerate_prompt.md` — course-issued, used during grading
- `grading/grade.py` — course-issued grader logic
- The package name `myproject` — pinned for grading reproducibility
- The Makefile target names — the TA runs them by name
- The directory layout (`src/`, `tests/`, `docs/`, `grading/`, `reports/`,
  `scripts/`) — the rubric paths are exact

## Where to start

The single most important file is `docs/SPEC.md`. The spec is the contract
between your team and the grader. The grader will feed it to an LLM and run
your user story tests against the generated code. Write the spec as if it
must be enough for someone to rebuild the project from scratch.

After SPEC.md, write `docs/STORIES.md`. Each story is graded twice: once by
your automated tests, once by the TA walking it manually. Stories must be
followable by a human with no source code access.

## Help

- Course rubric: `nlp_final_project_rubric_v10.docx`
- Office hours: see syllabus
- For technical issues with the template itself: open an issue on the
  course template repo, do NOT modify your team's copy
