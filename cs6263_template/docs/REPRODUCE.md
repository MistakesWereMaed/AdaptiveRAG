# Reproducibility Procedure

> The TA runs `make reproduce` to verify your headline numbers. This document
> tells the TA what to expect.

## Procedure

```bash
# From a fresh clone with .env populated
make reproduce
```

`make reproduce` performs these steps in order:

1. `make download-data` — fetches the datasets listed in `docs/DATA.md`
2. `make download-models` — fetches model checkpoints listed in `docs/MODELS.md`
3. Runs the application pipeline on a sample input
4. Runs `make test` (unit + integration + user story + edge)
5. Reports pass/fail per phase

## Hardware Profile

The headline numbers were measured on:

- CPU: Intel x86_64, 8 cores
- Memory: 16 GB
- Disk: 50 GB free
- Network: required for model and dataset downloads
- GPU: not required

## Expected Wall Clock

- Total `make reproduce` runtime: under 30 minutes on the documented hardware
- Of which `docker compose up` to healthy is under 10 minutes (Build category)
- Data and model download: 5 to 10 minutes depending on network
- Test suite: under 5 minutes

## Expected Outputs

After `make reproduce` completes, the following files exist:

- `reports/unit.xml` — unit test results
- `reports/integration.xml` — integration test results
- `reports/user_stories.xml` — user story acceptance test results
- `reports/edge.xml` — edge case test results
- `reports/coverage.xml` — coverage report
- `reports/coverage_html/index.html` — coverage browser

## Expected Metric Values

These are the headline numbers reported in `README.md`. The TA's reproduction
must match within the stated tolerance.

| Metric | Expected | Tolerance | Where measured |
|---|---|---|---|
| Accuracy on dev set | 0.85 | ± 0.02 | `reports/eval.json` |
| F1 score | 0.81 | ± 0.02 | `reports/eval.json` |
| p95 latency (single query) | 240 ms | ± 50 ms | `reports/benchmarks.json` |

## Outside Tolerance?

If a metric drifts outside the documented tolerance:

- The Reproducibility test row scores 5/10 instead of 10/10.
- The team is expected to investigate and document the cause in
  `reports/known_issues.md` if the deadline has not passed.
