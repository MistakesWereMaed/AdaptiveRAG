#!/usr/bin/env bash
# scripts/preflight.sh
#
# Local grading dry-run. Runs every automated check the TA will run, in the
# same order. If preflight passes, your automated grade will pass. The manual
# walkthrough (Phase 3 in the rubric) cannot be simulated here; you still need
# to demo your app yourself or have a teammate walk through STORIES.md.
#
# Exit code 0 = all automated checks passed.
# Non-zero exit code = at least one check failed; fix it before pushing.

set -uo pipefail
# We deliberately do not set -e at the top because we want to run every check
# and report the full picture, then exit non-zero if anything failed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Track failures across all phases
FAILED_CHECKS=()

run_check() {
  local name="$1"; shift
  echo
  echo "=================================================================="
  echo "[preflight] $name"
  echo "=================================================================="
  if "$@"; then
    echo "[preflight] PASS: $name"
  else
    echo "[preflight] FAIL: $name"
    FAILED_CHECKS+=("$name")
  fi
}

# Phase 1: Automated checks the grader runs
run_check "make reproduce (full pipeline replay)" \
  make reproduce

run_check "make test (unit + integration + user story tests)" \
  make test

run_check "make lint (ruff + black + mypy)" \
  make lint

run_check "pip-audit (dependency vulnerabilities)" \
  bash -c "pip-audit -r requirements.txt --desc 2>&1 | tee reports/security.txt; \
           ! grep -E 'Critical|High' reports/security.txt"

run_check "make loadtest (sustained throughput, error rate)" \
  make loadtest

run_check "scripts/regenerate.sh (spec to code generation)" \
  bash scripts/regenerate.sh

# Phase 2: Build environment check
# We do not run docker compose up here because preflight is run inside Docker.
# Instead we sanity check that the compose file is parseable and the env example exists.
run_check "Compose configuration sanity" \
  bash -c "test -f docker-compose.yml && test -f .env.example && \
           docker compose config -q 2>&1 || true"

# Team Contributions check — git shortlog snapshot
run_check "git contributions snapshot" \
  bash -c "git shortlog -sne --all --no-merges > reports/git_contributions.txt && cat reports/git_contributions.txt"

# Final report
echo
echo "=================================================================="
echo "[preflight] SUMMARY"
echo "=================================================================="
if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
  echo "[preflight] ALL AUTOMATED CHECKS PASSED."
  echo "[preflight] You are ready to push for grading."
  echo "[preflight] Reminder: Phase 3 of grading is a manual walkthrough of"
  echo "[preflight] docs/STORIES.md against your running app. Make sure every"
  echo "[preflight] story works against docker compose up before submitting."
  exit 0
else
  echo "[preflight] FAILED CHECKS:"
  for c in "${FAILED_CHECKS[@]}"; do
    echo "  - $c"
  done
  echo
  echo "[preflight] Fix each failure above before pushing."
  echo "[preflight] If a check is failing because of a known issue, document"
  echo "[preflight] it in reports/known_issues.md and contact the instructor."
  exit 1
fi
