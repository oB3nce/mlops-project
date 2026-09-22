# How this repo grows: HW1 → HW5

This is your semester project. You picked a dataset in the Week 2 *Define Project
Topic* milestone; each homework adds one MLOps capability to **this same repo**.
Full briefs live in the course materials under `homework/`.

| Milestone | Released | Week due | What you add to this repo |
| :--- | :--- | :--- | :--- |
| **Define Project Topic** | 2 | 2 | Add your dataset under `data/`, fill `docs/DATA_DICTIONARY.md`, write `docs/proposal.md`, set `.env`. Baseline runs. |
| **HW1** | 5 | 6 | DVC + MinIO data versioning; MLflow experiment tracking; register the first model version. |
| **HW2** | 7 | 8 | Pandera data validation; an evaluation gate with a documented go/no-go threshold. |
| **HW3** | 9 | 10 | A Prefect flow (prepare → validate → train → evaluate → register) with parameters + retries. |
| **HW4** | 11 | 12 | Serve the registered model with KServe; document a rollout; add smoke tests. |
| **HW5** | 13 | 14 | Prometheus + Grafana observability; an Evidently drift report; an alert + intervention plan. |

## Start here

```bash
cp .env.example .env          # then set DATA_PATH, TARGET_COLUMN, PROBLEM_TYPE
uv sync --all-groups
uv run pytest -v              # green on synthetic data, no dataset needed
# add your dataset to data/, then:
uv run python src/main.py     # baseline metrics on your data
```

Keep each homework's work in this repo's history (commits/PRs) — that history is
part of what's graded.
