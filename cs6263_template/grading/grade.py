#!/usr/bin/env python3
"""grading/grade.py

Compute the final rubric score from artifacts under reports/ and grading/.

Run from the repo root after the TA has executed every other check:

    python3 grading/grade.py

Output:
  * grading/score.json      machine readable score breakdown
  * grading/score.md        human readable summary the TA reviews

Exit code 0 always (the script reports the score; it does not pass/fail).

The categories and weights in this script must match the rubric exactly. If
the rubric changes, update CATEGORY_WEIGHTS and the per-category scoring
functions below.
"""

from __future__ import annotations

import json
import sys
import datetime
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
GRADING = REPO / "grading"

# Category weights must total 100 and match the rubric.
CATEGORY_WEIGHTS = {
    "spec_driven_development": 25,
    "reproducibility_manifest": 10,
    "build_and_deployment": 6,
    "verification_and_automated_testing": 12,
    "stress_and_robustness": 6,
    "code_quality_and_responsible_ai": 6,
    "logging": 5,
    "application_functionality_and_ui": 20,
    "user_documentation": 6,
    "team_contributions": 4,
}
assert sum(CATEGORY_WEIGHTS.values()) == 100, "weights must total 100"


@dataclass
class CategoryResult:
    name: str
    weight: int
    score: float
    notes: list[str] = field(default_factory=list)

    def as_md_row(self) -> str:
        notes = "; ".join(self.notes) if self.notes else ""
        return f"| {self.name.replace('_', ' ').title()} | {self.score:.1f} / {self.weight} | {notes} |"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_junit(path: Path) -> tuple[int, int, int, int]:
    """Return (total, passed, failed, skipped) from a JUnit XML file."""
    if not path.exists():
        return 0, 0, 0, 0
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return 0, 0, 0, 0
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    passed = total - failed - skipped
    return total, passed, failed, skipped


def file_exists(rel: str) -> bool:
    return (REPO / rel).exists()


def has_section(path: Path, heading: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="ignore").lower()
    return heading.lower() in text


# ---------------------------------------------------------------------------
# Per-category scoring
# Each function returns (score, notes).
# ---------------------------------------------------------------------------
def score_spec_driven_development() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["spec_driven_development"]
    notes = []

    # Prerequisites: missing any zeros the category
    prereqs = [
        "docs/SPEC.md",
        "docs/STORIES.md",
        "grading/traceability.yaml",
    ]
    missing = [p for p in prereqs if not file_exists(p)]
    if missing:
        notes.append(f"missing prerequisite(s): {', '.join(missing)}")
        return 0.0, notes

    # Test row: spec regeneration
    total, passed, _, _ = parse_junit(REPORTS / "regenerated_user_stories.xml")
    if total == 0:
        notes.append("regenerated_user_stories.xml absent or empty; spec regen did not run")
        return 0.0, notes

    ratio = passed / total
    if ratio < 0.50:
        notes.append(f"regen pass rate {ratio:.0%} < 50%; scores zero")
        return 0.0, notes
    if ratio >= 0.90:
        notes.append(f"regen pass rate {ratio:.0%} ≥ 90%; full credit")
        return float(weight), notes
    # Proportional between 50% and 90%
    score = weight * ratio
    notes.append(f"regen pass rate {ratio:.0%}; proportional score")
    return round(score * 2) / 2, notes  # round to nearest 0.5


def score_reproducibility_manifest() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["reproducibility_manifest"]
    notes = []
    prereqs = [
        "grading/manifest.yaml",
        "docs/DATA.md",
        "docs/MODELS.md",
        "docs/REPRODUCE.md",
    ]
    missing = [p for p in prereqs if not file_exists(p)]
    if missing:
        notes.append(f"missing prereq(s): {', '.join(missing)}")
        return 0.0, notes

    # The TA records reproduce result in reports/reproduce_status.txt:
    #   "full" | "partial" | "failed"
    status_file = REPORTS / "reproduce_status.txt"
    status = status_file.read_text().strip().lower() if status_file.exists() else ""
    if status == "full":
        notes.append("make reproduce: full credit (metrics within tolerance)")
        return float(weight), notes
    if status == "partial":
        notes.append("make reproduce: pipeline completed but metrics drifted")
        return weight / 2, notes
    notes.append("make reproduce: failed to complete or status not recorded")
    return 0.0, notes


def score_build_and_deployment() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["build_and_deployment"]
    notes = []
    prereqs = ["Dockerfile", "docker-compose.yml", ".env.example"]
    missing = [p for p in prereqs if not file_exists(p)]
    if missing:
        notes.append(f"missing prereq(s): {', '.join(missing)}")
        return 0.0, notes

    # The TA writes the build result to reports/build_status.txt:
    #   line 1: "ok" or "failed"
    #   line 2: integer count of undocumented manual steps
    status_file = REPORTS / "build_status.txt"
    if not status_file.exists():
        notes.append("build_status.txt not recorded by TA")
        return 0.0, notes
    lines = status_file.read_text().splitlines()
    state = lines[0].strip().lower() if lines else ""
    manual_steps = int(lines[1].strip()) if len(lines) > 1 and lines[1].strip().isdigit() else 0

    if state != "ok":
        notes.append("build did not reach healthy")
        return 0.0, notes
    score = max(0, weight - 1 * manual_steps)
    notes.append(f"build ok; {manual_steps} undocumented manual step(s); -{1 * manual_steps} pts")
    return float(score), notes


def score_verification_and_automated_testing() -> tuple[float, list[str]]:
    """Score breakdown matches the v1.4 rubric: 3 + 2 + 2 + 5 = 12 pts.

      * 3 pts for `make test` exit 0
      * 2 pts for all three JUnit XML reports parseable
      * 2 pts for coverage ≥ 70% on business logic
      * 5 pts for user story test pass rate (≥90% = full 5; proportional below)
    """
    weight = CATEGORY_WEIGHTS["verification_and_automated_testing"]
    notes = []
    score = 0.0

    # 3 pts for make test exit 0
    exit_file = REPORTS / "make_test_exit.txt"
    if exit_file.exists() and exit_file.read_text().strip() == "0":
        score += 3
        notes.append("make test exit 0: +3")
    else:
        notes.append("make test did not exit 0: +0")

    # 2 pts if all three JUnit XML files present and parseable
    junit_files = [
        REPORTS / "unit.xml",
        REPORTS / "integration.xml",
        REPORTS / "user_stories.xml",
    ]
    parseable = sum(1 for f in junit_files if parse_junit(f)[0] > 0)
    if parseable == 3:
        score += 2
        notes.append("all 3 JUnit reports parseable: +2")
    else:
        notes.append(f"only {parseable}/3 JUnit reports parseable: +0")

    # 2 pts for coverage at threshold
    cov_file = REPORTS / "coverage.xml"
    cov_pct = 0.0
    if cov_file.exists():
        try:
            root = ET.parse(cov_file).getroot()
            cov_pct = float(root.get("line-rate", 0)) * 100
        except (ET.ParseError, ValueError):
            cov_pct = 0.0
    if cov_pct >= 70:
        score += 2
        notes.append(f"coverage {cov_pct:.0f}% ≥ 70%: +2")
    else:
        notes.append(f"coverage {cov_pct:.0f}% < 70%: +0")

    # 5 pts for user story pass rate ≥90% (proportional below)
    total, passed, _, _ = parse_junit(REPORTS / "user_stories.xml")
    if total > 0:
        ratio = passed / total
        story_score = 5.0 if ratio >= 0.90 else 5 * ratio
        score += story_score
        notes.append(f"user story pass rate {ratio:.0%}: +{story_score:.1f}")
    else:
        notes.append("user_stories.xml absent or empty: +0")

    return round(min(score, weight) * 2) / 2, notes


def score_stress_and_robustness() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["stress_and_robustness"]
    notes = []
    score = 0.0

    # 4 pts: load test threshold met (TA records in reports/loadtest_status.txt)
    lt_status = REPORTS / "loadtest_status.txt"
    if lt_status.exists() and lt_status.read_text().strip().lower() == "ok":
        score += 4
        notes.append("load test ≥10 req/s, <5% errors: +4")
    else:
        notes.append("load test below threshold or not run: +0")

    # 2 pts: edge tests green
    total, passed, failed, _ = parse_junit(REPORTS / "edge.xml")
    if total > 0 and failed == 0:
        score += 2
        notes.append(f"edge tests green ({passed}/{total}): +2")
    elif total == 0:
        notes.append("edge.xml absent or empty: +0")
    else:
        notes.append(f"edge tests failed ({failed}/{total} failed): +0")
    return float(score), notes


def score_code_quality() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["code_quality_and_responsible_ai"]
    notes = []
    score = 0.0

    # 3 pts: make lint exit 0 (TA records in reports/lint_exit.txt)
    lint_file = REPORTS / "lint_exit.txt"
    if lint_file.exists() and lint_file.read_text().strip() == "0":
        score += 3
        notes.append("make lint exit 0: +3")
    else:
        notes.append("make lint did not exit 0: +0")

    # 1 pt: pip-audit clean of Critical/High
    sec_file = REPORTS / "security.txt"
    if sec_file.exists():
        text = sec_file.read_text()
        if "Critical" not in text and "High" not in text:
            score += 1
            notes.append("pip-audit no Critical/High: +1")
        else:
            notes.append("pip-audit flagged Critical/High: +0")
    else:
        notes.append("security.txt missing: +0")

    # 2 pts: model card has all four sections
    card = REPO / "docs" / "MODEL_CARD.md"
    sections = ["intended use", "limitations", "risks", "out of scope"]
    if card.exists():
        text = card.read_text().lower()
        present = sum(1 for s in sections if s in text)
        if present == 4:
            score += 2
            notes.append("model card all 4 sections: +2")
        else:
            notes.append(f"model card only {present}/4 sections: +0")
    else:
        notes.append("MODEL_CARD.md missing: +0")
    return float(score), notes


def score_logging() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["logging"]
    notes = []
    # The TA writes "complete" / "partial" / "missing" to reports/logging_trace.txt
    f = REPORTS / "logging_trace.txt"
    if not f.exists():
        notes.append("logging_trace.txt absent")
        return 0.0, notes
    state = f.read_text().strip().lower()
    if state == "complete":
        notes.append("trace complete: full credit")
        return float(weight), notes
    if state == "partial":
        notes.append("trace partial: half credit")
        return weight / 2, notes
    notes.append("trace missing or unstructured: zero")
    return 0.0, notes


def score_app_functionality() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["application_functionality_and_ui"]
    notes = []
    # The TA records each story result in reports/walkthrough.md following
    # the template format. We parse a simple machine-readable section the
    # TA fills out: each line "US-NN: pass" or "US-NN: fail".
    f = REPORTS / "walkthrough_results.txt"
    if not f.exists():
        notes.append("walkthrough_results.txt absent (TA fills during Phase 3)")
        return 0.0, notes
    lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
    total = sum(1 for l in lines if l.lower().endswith(("pass", "fail")))
    passed = sum(1 for l in lines if l.lower().endswith("pass"))
    if total == 0:
        notes.append("no story results found")
        return 0.0, notes
    ratio = passed / total
    score = round((weight * ratio) * 2) / 2
    notes.append(f"walkthrough {passed}/{total} passed ({ratio:.0%}); proportional: {score}")
    return score, notes


def score_user_documentation() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["user_documentation"]
    notes = []
    # The TA fills reports/docs_check.txt with three lines:
    #   line 1: cold-read result, integer 0..3 (3 = clean cold read)
    #   line 2: usage guide completeness, integer 0..2
    #   line 3: screenshot match, integer 0..1
    f = REPORTS / "docs_check.txt"
    if not f.exists():
        notes.append("docs_check.txt absent")
        return 0.0, notes
    lines = f.read_text().splitlines()
    try:
        cold = int(lines[0])
        usage = int(lines[1])
        screen = int(lines[2])
    except (IndexError, ValueError):
        notes.append("docs_check.txt malformed; expected 3 integer lines")
        return 0.0, notes
    score = max(0, min(3, cold)) + max(0, min(2, usage)) + max(0, min(1, screen))
    notes.append(f"cold-read {cold}/3, usage {usage}/2, screenshots {screen}/1")
    return float(score), notes


def score_team_contributions() -> tuple[float, list[str]]:
    weight = CATEGORY_WEIGHTS["team_contributions"]
    notes = []
    if not file_exists("CONTRIBUTIONS.md"):
        notes.append("CONTRIBUTIONS.md missing")
        return 0.0, notes
    if not file_exists("reports/git_contributions.txt"):
        notes.append("reports/git_contributions.txt missing")
        return 0.0, notes
    # Score is binary on two conditions; the TA records pass/fail in
    # reports/team_check.txt as two lines, "pass" or "fail" each:
    #   line 1: percentage match within ±15 points
    #   line 2: each member committed across at least 2 of src/tests/docs
    f = REPORTS / "team_check.txt"
    if not f.exists():
        notes.append("team_check.txt absent")
        return 0.0, notes
    lines = f.read_text().splitlines()
    pct = lines[0].strip().lower() == "pass" if lines else False
    breadth = lines[1].strip().lower() == "pass" if len(lines) > 1 else False
    score = (2 if pct else 0) + (2 if breadth else 0)
    notes.append(f"percentage match: {'pass' if pct else 'fail'}; breadth: {'pass' if breadth else 'fail'}")
    return float(score), notes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SCORERS = [
    ("spec_driven_development", score_spec_driven_development),
    ("reproducibility_manifest", score_reproducibility_manifest),
    ("build_and_deployment", score_build_and_deployment),
    ("verification_and_automated_testing", score_verification_and_automated_testing),
    ("stress_and_robustness", score_stress_and_robustness),
    ("code_quality_and_responsible_ai", score_code_quality),
    ("logging", score_logging),
    ("application_functionality_and_ui", score_app_functionality),
    ("user_documentation", score_user_documentation),
    ("team_contributions", score_team_contributions),
]


def main() -> int:
    GRADING.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    results: list[CategoryResult] = []

    for name, fn in SCORERS:
        score, notes = fn()
        results.append(CategoryResult(
            name=name, weight=CATEGORY_WEIGHTS[name], score=score, notes=notes,
        ))

    total_score = sum(r.score for r in results)
    total_weight = sum(r.weight for r in results)

    # JSON output
    payload = {
        "graded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total": total_score,
        "max": total_weight,
        "categories": [asdict(r) for r in results],
    }
    (GRADING / "score.json").write_text(json.dumps(payload, indent=2))

    # Markdown summary
    lines = [
        "# Final Project Grade",
        "",
        f"Graded at: {payload['graded_at']}",
        "",
        f"## Total: {total_score:.1f} / {total_weight}",
        "",
        "| Category | Score | Notes |",
        "|---|---|---|",
    ]
    lines.extend(r.as_md_row() for r in results)
    (GRADING / "score.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
