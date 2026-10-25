# HW2 — Data validation contracts

## R1 — Ingestion and model-input contracts

[`schemas.py`](../src/mlops_project/schemas.py) defines two distinct, strict
Pandera `DataFrameModel` contracts for the Telco churn dataset:

- `IngestionSchema` describes all 21 columns as they arrive from the raw CSV.
  It constrains the categorical domains, requires unique customer IDs, and
  bounds each numeric column: `SeniorCitizen` 0–1, `tenure` 0–120 months, and
  `MonthlyCharges` $0–$500. `TotalCharges` arrives as text in this CSV, so the
  ingestion contract verifies that values are numeric text or blank before
  preprocessing.
- `ModelInputSchema` describes the 19 feature columns passed to the model.
  It requires `TotalCharges` to be numeric (0–$60,000 when present) and
  preserves nulls for its known blank-string records; the model pipeline
  imputes those values. `customerID` and the target are not part of model
  input. **Customer IDs are unique record identifiers, not predictive
  features, so excluding them prevents the model from memorizing individual
  customers.**

The bounds are deliberately broader than the observed ranges so plausible
records remain valid while unit errors or corrupted values are caught. In the
current 7,043-row snapshot, `SeniorCitizen` is 0–1, `tenure` is 0–72,
`MonthlyCharges` is $18.25–$118.75, and the nonblank `TotalCharges` values are
within $0–$8,684.80. There are 11 blank `TotalCharges` entries; they are
expected source values for new accounts and are retained as missing values for
the training pipeline's imputer. No known impossible values occur in this
snapshot, but the declared range and categorical constraints protect against
corruption in future versions.

The prepare stage validates the raw CSV with `IngestionSchema` immediately
after reading it. It then converts numeric text, removes the customer ID, and
validates the resulting feature frame with `ModelInputSchema` before splitting
the data. `dvc.yaml` tracks `schemas.py` as a prepare-stage dependency.

## R2 — Validation gate in the DVC pipeline

The `validate` stage runs `IngestionSchema` against `data/raw/dataset.csv` and
writes its pass/fail verdict to `reports/validation_report.json`. This is a
`cache: false` output so the current validation result is available in the
workspace. The `prepare` stage declares that report as a dependency; this
validate-to-prepare edge makes the verdict a gate. A failed contract writes a
failure report and exits with status 1, so DVC stops before prepare, train, or
evaluate.

To safely demonstrate the gate, run:

```powershell
$env:PYTHONUTF8 = "1"
uv run python -m mlops_project.demo_validation_gate
```

The script saves the original dataset bytes, temporarily sets the first
`MonthlyCharges` value to `-1`, runs `dvc repro`, and restores the exact
original bytes in a `finally` block. It succeeds only if DVC fails at the
`validate` stage as expected. The following output was captured:

```text
Temporarily changed the first MonthlyCharges value to -1.
Restored the original dataset.
Running stage 'validate':
> uv run python -m mlops_project.validate
Data validation FAILED. Stopping pipeline.
ERROR: failed to reproduce 'validate': failed to run: uv run python -m mlops_project.validate, exited with 1
Confirmed: dvc repro stopped at the ingestion validation gate.
Restored dataset MD5:
0F9DE68E012BD3AED5FA7CDC9FC421AF
```

The invalid-data verdict contains the failed `MonthlyCharges >= 0` check:

```json
{
  "contract": "IngestionSchema",
  "error_count": 1,
  "n_rows": 7043,
  "status": "fail"
}
```

After the demo restores the data, run `uv run dvc repro` to record the passing
verdict again. This was verified: validation passed for all 7043 rows, the
remaining stages were skipped, and DVC reported that the pipeline was up to
date.

## R3/R4/R8 — Slice metrics, constant baseline, and fairness gap

The `evaluate` section of [`params.yaml`](../params.yaml) declares two
stakeholder-relevant slices:

- **Contract type** (`Contract`): month-to-month customers may have less
  commitment and different churn risk than one- or two-year customers. This
  helps retention teams compare model quality for customers with different
  contract commitments.
- **Senior-citizen status** (`SeniorCitizen`): the groups are non-senior (`0`)
  and senior (`1`). This lets the team compare churn predictions and true
  positive rates for an age-related customer group.

`evaluate.py` builds these slices from `data/processed/test_raw_slices.csv`,
which is written from the held-out rows before feature preprocessing. Each
classification slice row contains `n`, `positive_count` (the actual `Churn =
"Yes"` count), and accuracy, precision, recall, F1, selection rate, and TPR.
Empty groups have null metric values; missing-valued records do not match any
named group and are excluded from those groups.

The model pipeline median-imputes numeric `TotalCharges` values. There are 11
blank values in the full source data. They are not a configured slice, so they
do not change membership in the contract or senior-citizen groups. If a slice
were made on `TotalCharges` after imputation, those blanks would be replaced
with the training median and could no longer be counted as missing; their
group membership and the resulting per-slice metrics would change. Using the
raw held-out rows preserves that distinction.

The aggregate report also includes a constant baseline. For this
classification task it predicts the majority class from the training labels
only (not the test labels), then reports its test metrics. The fairness slice
is `senior_citizen`; its selection-rate gap is
`max(selection_rate) - min(selection_rate)` and its TPR gap is
`max(tpr) - min(tpr)` across the configured groups. The generated results are
written to `metrics/metrics.json` and `metrics/slices.json`; both are DVC
metrics with `cache: false`.

On the current held-out set, the model achieved accuracy **0.8020** and
weighted F1 **0.7939**. The training-majority baseline predicts `No` and
achieved accuracy **0.7346** and weighted F1 **0.6222**. The generated slice
results were:

| Slice group | n | Positive (`Yes`) | Accuracy | F1 | Selection rate | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contract: month-to-month | 773 | 329 | 0.6986 | 0.6958 | 0.3803 | 0.5927 |
| Contract: one year | 300 | 36 | 0.8767 | 0.8222 | 0.0033 | 0.0000 |
| Contract: two year | 336 | 9 | 0.9732 | 0.9600 | 0.0000 | 0.0000 |
| SeniorCitizen: non-senior | 1187 | 276 | 0.8222 | 0.8125 | 0.1727 | 0.4891 |
| SeniorCitizen: senior | 222 | 98 | 0.6937 | 0.6920 | 0.4054 | 0.6122 |

For the `senior_citizen` fairness slice, the observed selection-rate gap is
**0.2327** and the TPR gap is **0.1231**. The one-year and two-year contract
groups have only 36 and 9 positive examples respectively, so their TPR values
should be interpreted cautiously.

## R5/R7/R10 — Acceptance criteria and champion baseline

`acceptance` in [`params.yaml`](../params.yaml) is versioned as criteria
version `1`. The thresholds are derived from the current held-out evaluation,
not copied from the lab:

| Criterion | Configured value | Measured project result and derivation |
| --- | ---: | --- |
| Minimum accuracy | 0.79 | Current model accuracy is 0.8020, so the floor allows a 0.0120 decrease. The constant `No` predictor scores 0.7346; this floor is 0.0554 above it. |
| Minimum weighted F1 | 0.78 | Current model F1 is 0.7939, so the floor allows a 0.0139 decrease. The constant predictor scores 0.6222 F1; this floor is 0.1578 above it. |
| Regression primary metric | F1, tolerance 0.01 | The current champion is at 0.7939 F1. A 0.01 absolute tolerance allows at most a one-point decrease (minimum 0.7839), which is smaller than the measured 0.0139 space between the current F1 and its absolute floor. |
| Slice F1 floor | 0.65 | The lowest measured group F1 is 0.6920 for senior customers. The rule leaves 0.0420 headroom below that observed weakest group while still rejecting a material loss in group quality. |
| Slice minimum support | 200 | The smallest configured group currently has 222 rows (senior customers); 200 is 22 below that observed minimum, tolerating modest sample variation while rejecting very small groups. |

The two absolute floors both exceed the measured constant-predictor score for
the same metric. In particular, the accuracy floor is above baseline accuracy
by 0.0554, and the F1 floor is above baseline F1 by 0.1578. Slice rules apply
the configured F1 floor and minimum support to every evaluated group for both
the `contract` and `senior_citizen` slices.

The acceptance logic is a pure `evaluate_acceptance(...)` function in
[`gate.py`](../src/mlops_project/gate.py): it receives ordinary dictionaries,
does not read files or access MLflow, and returns per-check results plus an
overall verdict. The DVC `gate` stage reads the evaluated JSON files, optional
`metrics/baseline.json`, and acceptance parameters, then writes
`reports/gate.json` with `cache: false`. Its verdict is `GO`, `NO-GO`, or
`SKIPPED`. The baseline file is deliberately not a DVC dependency so the
no-baseline path can execute. A failed criterion exits non-zero. If the
baseline is absent, only regression is marked `skipped`, and the overall
verdict is `SKIPPED`, never `GO`.

The separate publication command is
[`register_model.py`](../src/mlops_project/register_model.py), not the `train`
stage. Before making any MLflow Registry calls it reads `reports/gate.json` and
requires the exact verdict `GO` and a `run_id` matching the candidate being
registered. `NO-GO`, `SKIPPED`, stale, missing, or malformed verdicts stop
registration and alias promotion.

### Refreshing the committed baseline by MLflow alias

With Compose services running, the current run can be assigned the `champion`
alias and the committed baseline file refreshed from that alias:

```powershell
uv run python -m mlops_project.register_model --alias champion
uv run python -m mlops_project.refresh_baseline
uv run dvc repro gate --single-item --force
```

The refresh script resolves the registered model using
`get_model_version_by_alias(model_name, "champion")`; it does not select a
version number. It then reads the resolved run's `model_*` metrics and writes
their values and alias/run provenance to `metrics/baseline.json`. The
registered model name defaults to `telco-churn-model` and can be overridden
with `MLFLOW_REGISTERED_MODEL_NAME` in `.env`. The current refresh resolved
`telco-churn-model@champion`, run
`3c2af3aeb423436eb64482c4ffc70c59`, with accuracy `0.8020` and F1 `0.7939`.
Commit `metrics/baseline.json` with the project changes so a fresh checkout
has the champion reference. After refreshing it, force the single gate stage
as shown above because the optional baseline is intentionally not a DVC
dependency.

The missing-baseline path was exercised by temporarily moving the baseline
file, running the gate stage alone, then restoring the file and rerunning the
gate. The relevant report was:

```json
{
  "name": "regression.f1",
  "reason": "No champion baseline file is available.",
  "status": "skipped"
}
```

The overall verdict in that run was `"SKIPPED"`; after restoring the baseline,
the gate reported `"GO"`. The three focused tests in
[`test_gate.py`](../tests/test_gate.py) cover pass, fail, and missing-baseline
behavior.

## R6 — Publication is blocked unless the gate is GO

The pipeline's `train` stage only writes `models/model.pkl` and the MLflow run
ID; it does not register or promote a model. The separate registration command
first checks `reports/gate.json` and refuses unless `verdict` is exactly
`"GO"`. The failure demonstration changed only `max_depth` from `10` to `1`, producing
a deliberately shallow model. Its actual held-out accuracy and F1 fell to
`0.7346` and `0.6222`; the month-to-month and senior slice F1 scores fell to
`0.4191` and `0.4004`.

```powershell
$env:PYTHONUTF8 = "1"
uv run dvc repro
Test-Path models/model.pkl
uv run python -m mlops_project.register_model --alias staging
```

Captured command output:

```text
Trained random_forest model; MLflow run ID: 8d90ab92ca654869aa9a6fcadc1b065d
Aggregate metrics: {'model': {'accuracy': 0.7346, 'precision': 0.5396, 'recall': 0.7346, 'f1': 0.6222}, ...}
Running stage 'gate':
  "verdict": "NO-GO"
ERROR: failed to reproduce 'gate': failed to run: uv run python -m mlops_project.gate, exited with 1

Test-Path models\model.pkl
True

uv run python -m mlops_project.register_model --alias staging
RuntimeError: Cannot publish model: gate verdict is 'NO-GO'; expected 'GO'.
```

The training output and file check confirm `models/model.pkl` still existed
(126,978 bytes) after DVC stopped at the gate. Registration refused before
making any MLflow Registry calls. The degraded run was not promoted. The
hyperparameter was restored to `max_depth: 10`, then the full pipeline was
rerun and `reports/gate.json` returned to `"GO"`.
