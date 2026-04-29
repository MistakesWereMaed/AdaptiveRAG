#!/usr/bin/env bash
# scripts/regenerate.sh
#
# Spec regeneration test runner.
# Feeds docs/SPEC.md through the course-issued prompt template to the pinned LLM,
# parses the file blocks back into a clean regenerated/ tree, and runs the
# user story acceptance tests against the regenerated code.
#
# Usage: bash scripts/regenerate.sh
# Requires: ANTHROPIC_API_KEY in the environment, and Python with the dependencies
#           in requirements.txt installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pinned configuration. The course requires these exact values for grading
# reproducibility. Do not change them.
MODEL="claude-opus-4-5-20251101"
TEMPERATURE="0"
MAX_TOKENS="64000"
PROMPT_TEMPLATE="scripts/regenerate_prompt.md"
SPEC="docs/SPEC.md"
OUTPUT_DIR="regenerated"

# Sanity checks
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set in the environment." >&2
  echo "Set it with: export ANTHROPIC_API_KEY=sk-ant-..." >&2
  exit 1
fi

for f in "$PROMPT_TEMPLATE" "$SPEC"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file missing: $f" >&2
    exit 2
  fi
done

# Clean and recreate the regenerated/ tree
echo "[regen] cleaning $OUTPUT_DIR/"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/src/myproject"

# Step 1. Build the prompt by substituting the spec into the template
echo "[regen] building prompt from $PROMPT_TEMPLATE + $SPEC"
PROMPT_FILE="$(mktemp)"
trap 'rm -f "$PROMPT_FILE"' EXIT

python3 - "$PROMPT_TEMPLATE" "$SPEC" "$PROMPT_FILE" <<'PYEOF'
import sys, pathlib
template_path, spec_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
template = pathlib.Path(template_path).read_text()
spec = pathlib.Path(spec_path).read_text()
prompt = template.replace("{{SPEC_CONTENT}}", spec)
pathlib.Path(out_path).write_text(prompt)
PYEOF

# Step 2. Call the Anthropic API with pinned config
echo "[regen] calling Anthropic API (model=$MODEL temperature=$TEMPERATURE)"
RESPONSE_FILE="$OUTPUT_DIR/api_response.json"

python3 - "$PROMPT_FILE" "$RESPONSE_FILE" "$MODEL" "$TEMPERATURE" "$MAX_TOKENS" <<'PYEOF'
import sys, os, json, urllib.request

prompt_path, response_path, model, temperature, max_tokens = sys.argv[1:6]
prompt = open(prompt_path).read()

body = json.dumps({
    "model": model,
    "max_tokens": int(max_tokens),
    "temperature": float(temperature),
    "messages": [{"role": "user", "content": prompt}],
}).encode()

req = urllib.request.Request(
    "https://api.anthropic.com/v1/messages",
    data=body,
    headers={
        "Content-Type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    },
)
with urllib.request.urlopen(req, timeout=600) as r:
    open(response_path, "wb").write(r.read())
PYEOF

# Step 3. Parse the file blocks out of the response into regenerated/src/...
echo "[regen] parsing file blocks into $OUTPUT_DIR/"
python3 - "$RESPONSE_FILE" "$OUTPUT_DIR" <<'PYEOF'
import sys, json, pathlib, re

response_path, out_dir = sys.argv[1], sys.argv[2]
data = json.load(open(response_path))

# Concatenate text blocks from the response
text = "".join(block["text"] for block in data["content"] if block["type"] == "text")

# Match === FILE: <path> === ... === END FILE ===
pattern = re.compile(
    r"=== FILE:\s*(?P<path>[^\s=]+)\s*===\s*\n(?P<body>.*?)\n=== END FILE ===",
    re.DOTALL,
)
matches = list(pattern.finditer(text))
if not matches:
    print("ERROR: no file blocks found in API response", file=sys.stderr)
    print("Response text begins:", file=sys.stderr)
    print(text[:500], file=sys.stderr)
    sys.exit(3)

count = 0
for m in matches:
    rel = m.group("path").strip()
    # Refuse path traversal
    if rel.startswith("/") or ".." in rel.split("/"):
        print(f"WARN: skipping suspicious path {rel}", file=sys.stderr)
        continue
    target = pathlib.Path(out_dir) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(m.group("body"))
    count += 1
    print(f"  wrote {target}")

if count == 0:
    print("ERROR: parser found blocks but none wrote a file", file=sys.stderr)
    sys.exit(3)
print(f"[regen] wrote {count} file(s)")
PYEOF

# Step 4. Run the user story tests against the regenerated source
echo "[regen] running user story tests against regenerated code"
mkdir -p reports
# We point pytest at the regenerated source by placing its src on the path.
PYTHONPATH="$OUTPUT_DIR/src:$PYTHONPATH" \
  pytest tests/user_stories/ \
    --junitxml=reports/regenerated_user_stories.xml \
    -p no:cacheprovider \
    || true   # do not abort on test failures; the grader counts pass ratio

# Step 5. Summarize
echo "[regen] summarizing results into reports/regeneration.md"
python3 - "reports/regenerated_user_stories.xml" "reports/regeneration.md" <<'PYEOF'
import sys, xml.etree.ElementTree as ET, pathlib, datetime

xml_path, md_path = sys.argv[1], sys.argv[2]
try:
    root = ET.parse(xml_path).getroot()
except (FileNotFoundError, ET.ParseError) as e:
    pathlib.Path(md_path).write_text(
        f"# Regeneration Report\n\nGeneration produced no parseable test results.\nError: {e}\n"
    )
    sys.exit(0)

# JUnit XML: testsuite[s] element with attributes tests, failures, errors, skipped
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
total = sum(int(s.get("tests", 0)) for s in suites)
failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
passed = total - failed - skipped
ratio = (passed / total) if total else 0.0

md = f"""# Spec Regeneration Report

Generated: {datetime.datetime.utcnow().isoformat()}Z

| Metric | Value |
|---|---|
| Total user story tests | {total} |
| Passed on regenerated code | {passed} |
| Failed | {failed} |
| Skipped | {skipped} |
| Pass ratio | {ratio:.1%} |

Rubric thresholds: ≥90% earns full credit; <50% scores zero. Between is proportional.

See reports/regenerated_user_stories.xml for per-test detail.
"""
pathlib.Path(md_path).write_text(md)
print(md)
PYEOF

echo "[regen] done. See reports/regeneration.md"
