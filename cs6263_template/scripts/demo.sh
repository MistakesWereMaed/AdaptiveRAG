#!/usr/bin/env bash
# scripts/demo.sh
#
# End-to-end demo. Exercises every user story against the running app and
# prints pass / fail per story. Intended to be run while `docker compose up`
# is running in another terminal.
#
# This is NOT the same as the user story acceptance tests in tests/user_stories/.
# Those are pytest tests run during `make test`. This demo hits the live
# application from the outside, the same way the TA will during the manual
# walkthrough, and gives you a quick health check.

set -uo pipefail

APP_URL="${APP_URL:-http://localhost:8080}"
PASSED=0
FAILED=0
TOTAL=0

# Confirm the app is reachable before we start
if ! curl -sf -o /dev/null --max-time 5 "$APP_URL/health"; then
  echo "ERROR: app is not reachable at $APP_URL/health" >&2
  echo "Start the app first: docker compose up" >&2
  exit 1
fi

echo "=================================================================="
echo "Demo run against $APP_URL"
echo "=================================================================="

run_story() {
  local id="$1"
  local description="$2"
  local cmd="$3"
  TOTAL=$((TOTAL + 1))
  echo
  echo "--- $id: $description ---"
  if eval "$cmd"; then
    echo "[$id] PASS"
    PASSED=$((PASSED + 1))
  else
    echo "[$id] FAIL"
    FAILED=$((FAILED + 1))
  fi
}

# Add one run_story line per user story in docs/STORIES.md.
# Replace the example below with your team's stories.

run_story "US-01" \
  "User can submit a query and receive a response" \
  "curl -sf -X POST $APP_URL/api/query -H 'Content-Type: application/json' \
        -d '{\"text\": \"hello\"}' | grep -q response"

run_story "US-02" \
  "User receives an error message for empty input" \
  "curl -s -X POST $APP_URL/api/query -H 'Content-Type: application/json' \
        -d '{\"text\": \"\"}' | grep -qi error"

# Add more run_story calls here for every story in your STORIES.md

echo
echo "=================================================================="
echo "Demo summary: $PASSED / $TOTAL stories passed ($FAILED failed)"
echo "=================================================================="

[[ $FAILED -eq 0 ]]
