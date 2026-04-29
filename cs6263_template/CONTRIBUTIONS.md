# Team Contributions

| Member | Role | Modules / Areas Owned | Percent |
|---|---|---|---|
| [Name 1] | [e.g. Backend lead] | src/myproject/router.py, src/myproject/retriever.py | 30% |
| [Name 2] | [e.g. Frontend lead] | src/myproject/api.py, docs/usage.md | 25% |
| [Name 3] | [e.g. ML lead] | src/myproject/generator.py, docs/MODEL_CARD.md | 25% |
| [Name 4] | [e.g. Ops lead] | Dockerfile, docker-compose.yml, scripts/ | 20% |

Total must sum to 100%.

## Verification

Run `git shortlog -sne --all --no-merges` and write the output to
`reports/git_contributions.txt`. The TA compares this distribution to the
table above; each member's commit share must be within ±15 percentage points
of declared share.

Each member must have commits across at least two of: `src/`, `tests/`, `docs/`.

```bash
git shortlog -sne --all --no-merges > reports/git_contributions.txt
```
