# Usage Guide

> Every feature listed in `docs/STORIES.md` must have a corresponding section here.
> The TA verifies this mapping during the Documentation walkthrough.

## Submitting a query (US-01)

To ask a question:

1. Visit http://localhost:8080.
2. Type your question in the search box.
3. Click "Submit".
4. The answer appears below the search box, with citations.

Tips:
- Questions phrased as full sentences work better than keyword fragments.
- The system retrieves the top 5 most relevant documents by default. To change,
  pass `max_results` in the API request body (see `docs/SPEC.md` section 4.1).

## Empty input handling (US-02)

If you click Submit without typing anything, the UI displays
"Please enter a question" inline. No API call is made, so no quota is consumed.

## Configuration troubleshooting (US-03)

If the system shows "The model service is not configured", the
`ANTHROPIC_API_KEY` is missing or empty in `.env`. Stop the app, edit `.env`,
and run `docker compose up` again.

> Add one section per story.
