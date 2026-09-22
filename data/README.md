# data/

Put your project dataset here (e.g. `data/dataset.csv`) and set `DATA_PATH` in
`.env`. Until then, `uv run python src/main.py` will tell you the file is
missing — that's expected. The tests run on synthetic data, so `pytest` is green
without a dataset.

You may commit the CSV here for now if its size and license allow. From **HW1
(Week 5)** you will version it with **DVC + MinIO** instead of committing it to
git — at that point this folder holds a `.dvc` pointer, not the raw file.

Document your data in `docs/DATA_DICTIONARY.md` before you start building.
