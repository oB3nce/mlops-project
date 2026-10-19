# My MLOps Project

Your semester-long project for **Lifecycle of Artificial Intelligence Systems**.
You start from this minimal, working baseline and grow it into a full MLOps
project across HW1–HW5 — see [`HOMEWORK.md`](HOMEWORK.md).

This repo is **separate from the course materials**. The course repo (lectures,
labs, datasets, homework briefs) you read and do the labs in; *this* repo is your
own project, on your own dataset, and is what gets graded.

## What's here

```
src/mlops_project/   config.py · data.py · model.py · cli.py   (a simple baseline pipeline)
src/main.py          entry point
tests/               smoke tests that pass on synthetic data (no dataset needed)
data/                put your dataset here (see data/README.md)
docs/                DATA_DICTIONARY.md (fill it in) + your proposal
.env.example         configuration — copy to .env
```

The baseline trains a scikit-learn `StandardScaler` + (logistic or linear)
regression on a CSV you point it at — classification or regression, set by
`PROBLEM_TYPE`. It's deliberately simple; the course is about the lifecycle, not
the model.

## Quick start

```bash
cp .env.example .env          # set DATA_PATH, TARGET_COLUMN, PROBLEM_TYPE
uv sync --all-groups
uv run pytest -v              # green immediately (synthetic data)
# drop your dataset in data/ and:
uv run python src/main.py     # baseline metrics on your data
```

Prerequisites: `uv` (https://docs.astral.sh/uv/) and Python 3.12+ (uv installs it).

## Local tracking services

The Docker Compose stack starts Postgres for MLflow's metadata, Silo for model
artifacts, and the MLflow tracking server. Copy `.env.example` to `.env` and
adjust the credentials there; `.env` is git-ignored and must not be committed.
The committed example contains placeholders.

```bash
docker compose up -d --wait
```

Open the MLflow UI at <http://localhost:5500>. The host ports map to these
container ports:

| Service | Host port | Container port | Purpose |
| --- | ---: | ---: | --- |
| MLflow | 5500 | 5000 | Tracking UI and API |
| Silo | 5510 | 9000 | S3-compatible API |
| Silo | 5511 | 9001 | Silo console |
| Postgres | 5532 | 5432 | MLflow metadata database |

These host ports are configurable in `.env`. The stack uses the `55xx` port
block from the course labs, so only one stack using those ports can run on this
host at a time. Stop the other stack first, or change these host ports to
unused values. Compose also has an explicit project name so its containers and
volumes do not collide with course lab stacks.

Stop the services with `docker compose down`; this preserves the named data
volumes. Use `docker compose down -v` only when you also want to delete the
stored database and artifacts.

## Next

1. Add your dataset to `data/` and document it in `docs/DATA_DICTIONARY.md`.
2. Write your one-page proposal in `docs/proposal.md` (Week 2 milestone).
3. Follow [`HOMEWORK.md`](HOMEWORK.md) from HW1 onward.
