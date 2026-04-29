# User Stories

> Every story below has a stable ID, a Given/When/Then statement, and numbered
> manual steps the TA can follow against the live `docker compose up` system.
> The TA scores Application Functionality (20 points) by walking these stories
> against the live UI in Phase 3 of grading.
>
> Format conventions:
>   * Story IDs are US-NN (US-01, US-02, ...). Filenames lowercase the prefix
>     and use underscores: US-01 maps to `test_us_01.py` and `us_01_expected.png`.
>   * Every story has a corresponding test in `tests/user_stories/`.
>   * Every story has a reference screenshot in `docs/assets/stories/`.
>   * Stories that exercise error paths are marked with [ERROR PATH] in the title.
>     At least 2 stories must be error path stories per the rubric.

---

## US-01: User submits a query and receives a cited answer

**As a** researcher
**I want** to submit a question and receive an answer with citations
**So that** I can trust where the information came from.

**Acceptance criteria (Given / When / Then):**

> Given the application is running and the corpus is indexed,
> When the user submits the query "What is FIPS 140-3?",
> Then the response contains a non-empty answer string and at least one citation
> with a doc_id and snippet, and the latency is under 3 seconds.

**Manual walkthrough steps:**

1. Confirm the app is running by visiting http://localhost:8080. The home page
   shows a search box and a "Submit" button.
2. Type "What is FIPS 140-3?" into the search box.
3. Click "Submit".
4. Observe the response area below the search box. It populates within 3 seconds.
5. Verify that the response contains:
   (a) an answer paragraph (non-empty),
   (b) a "Citations" subsection with at least one entry,
   (c) each citation showing a document title and a quoted snippet.
6. Compare the screen to `docs/assets/stories/us_01_expected.png`. The layout
   should match (exact text content will vary).

**Expected end state:** see `docs/assets/stories/us_01_expected.png`.

---

## US-02 [ERROR PATH]: Empty input shows an actionable error message

**As a** user
**I want** clear feedback when I submit an empty query
**So that** I know what to fix.

**Acceptance criteria (Given / When / Then):**

> Given the application is running,
> When the user clicks Submit without typing anything,
> Then an inline error message appears stating "Please enter a question",
> and no API call is made.

**Manual walkthrough steps:**

1. Confirm the app is running at http://localhost:8080.
2. Leave the search box empty.
3. Click "Submit".
4. Observe the error message that appears next to the search box: "Please enter a question".
5. Verify the response area below the search box is unchanged from the
   previous state (no spinner, no stack trace, no blank answer).
6. Open browser dev tools, Network tab, confirm no request was sent to /api/query.
7. Compare to `docs/assets/stories/us_02_expected.png`.

**Expected end state:** see `docs/assets/stories/us_02_expected.png`.

---

## US-03 [ERROR PATH]: Missing API key surfaces a clear server error

**As an** operator
**I want** to know when an upstream LLM credential is missing
**So that** I can fix configuration without reading logs.

**Acceptance criteria (Given / When / Then):**

> Given the application is running with an empty ANTHROPIC_API_KEY,
> When the user submits any non-empty query,
> Then the response shows "The model service is not configured. Contact the operator."
> and the HTTP status is 503, not 500.

**Manual walkthrough steps:**

1. Stop the running application: Ctrl-C in the docker compose terminal.
2. Edit `.env` and set `ANTHROPIC_API_KEY=` (empty).
3. Start the application: `docker compose up`.
4. Visit http://localhost:8080.
5. Type any non-empty query, e.g. "hello", and click Submit.
6. Observe the error message: "The model service is not configured. Contact the operator."
7. In dev tools Network tab, confirm the response status is 503.
8. Confirm no Python stack trace appears anywhere in the UI.
9. Compare to `docs/assets/stories/us_03_expected.png`.

**Expected end state:** see `docs/assets/stories/us_03_expected.png`.

---

> Add stories US-04, US-05, ... here for every major feature in the spec.
> Repeat the format above. Each story must have:
>   - A stable ID
>   - Given / When / Then
>   - Numbered manual steps a human can follow without reading source code
>   - A reference screenshot in docs/assets/stories/
>   - A matching test in tests/user_stories/test_us_NN.py
