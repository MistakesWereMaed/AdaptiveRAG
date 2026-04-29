# Grading Walkthrough Log

> This file is filled in by the TA during Phase 2 and Phase 3 of grading.
> The grading script reads structured snippets from this log to compute scores
> for App Functionality, Logging, and Documentation. Keep the section headers
> and the per-story result lines in their declared format.

Team: __________
Repo URL: __________
Commit SHA graded: __________
TA: __________
Graded at: __________ (UTC)

---

## Phase 2 — Build environment and bring up the system

### Build status

[ ] `docker compose up` reached healthy with no manual intervention.
[ ] One or more undocumented manual steps were required (count: ___).
[ ] Build did not reach healthy.

Wall clock from `docker compose up` to all services healthy: ___ minutes.

Notes (one line per undocumented manual step the TA had to discover):

- ...

> The TA writes a two-line file to `reports/build_status.txt`:
> ```
> ok           # or "failed"
> 0            # integer count of undocumented manual steps
> ```

---

### Logging trace

The TA picks one user-initiated action (UI click or API call) and traces it
end-to-end through `docker compose logs -f app` using its request ID.

Picked action: [e.g. "Submitted query 'hello' via UI at 14:32:01"]

Result:

[ ] **complete** — every component the request touched produced a log line
    tagged with the same request ID; full request lifecycle visible from logs alone.
[ ] **partial** — some components logged the request ID, others did not.
[ ] **missing** — request ID not found in logs, or logs unstructured.

> The TA writes the chosen word to `reports/logging_trace.txt` (single line:
> "complete" or "partial" or "missing").

---

## Phase 3 — Manual walkthrough against the running app

### README cold-read (Documentation)

The TA reads ONLY `README.md` (no other docs, no Slack, no email) and follows
the quick start cold from a fresh clone.

Result:

[ ] **3** — quick start worked exactly as written; reached running app.
[ ] **2** — reached running app, with one undocumented step the TA had to discover.
[ ] **1** — reached running app, with two undocumented steps.
[ ] **0** — could not reach a running app from the README alone.

Undocumented steps the TA had to discover:

- ...

### Usage guide completeness (Documentation)

The TA opens `docs/usage.md` and verifies every feature listed in `STORIES.md`
has a corresponding usage section.

Result:

[ ] **2** — every story has a usage section.
[ ] **1** — most stories have usage sections; 1 to 2 missing.
[ ] **0** — many stories have no usage section, or `docs/usage.md` is absent.

Missing sections:

- ...

### Screenshot match (Documentation)

The TA compares each `us_NN_expected.png` to the live UI during the story walkthrough.

Result:

[ ] **1** — screenshots match the live UI in layout and content.
[ ] **0** — screenshots are missing, stale, or do not match.

> The TA writes `reports/docs_check.txt`, three lines, one integer per line:
> ```
> 3        # cold-read score, 0..3
> 2        # usage guide completeness, 0..2
> 1        # screenshot match, 0..1
> ```

---

### Per-story walkthrough

For each story in `docs/STORIES.md`, the TA follows the numbered manual steps
and records the outcome here. A story passes only if every numbered step
behaves exactly as written and the end state matches the reference screenshot.

#### US-01: [Story title]

Result: [ ] PASS  [ ] FAIL

Observation (one line):
- ...

#### US-02: [Story title]

Result: [ ] PASS  [ ] FAIL

Observation (one line):
- ...

[continue for every story in STORIES.md]

> The TA writes `reports/walkthrough_results.txt`, one line per story:
> ```
> US-01: pass
> US-02: pass
> US-03: fail
> US-04: pass
> ...
> ```
> The grading script reads this file and computes
> (passed / total) × 20 for App Functionality.

---

## Team Contributions check

The TA runs `git shortlog -sne --all --no-merges` and compares to `CONTRIBUTIONS.md`.

Each member's commit share within ±15 percentage points of declared:
[ ] pass  [ ] fail

Each member committed across at least 2 of: src/, tests/, docs/:
[ ] pass  [ ] fail

> The TA writes `reports/team_check.txt`, two lines:
> ```
> pass     # or "fail"
> pass     # or "fail"
> ```

---

## TA final notes

[Free-form observations, exceptional cases, or context the grading script
cannot capture. Optional but encouraged.]
