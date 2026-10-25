from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from mlops_project.prepare import _normalize_columns
from mlops_project.schemas import IngestionSchema, ModelInputSchema


def _incoming_telco_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-ABCDE", "0002-BCDEF"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 72],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 118.75],
            "TotalCharges": pd.Series(["", "8543.25"], dtype="string"),
            "Churn": ["No", "Yes"],
        }
    )


def test_ingestion_schema_accepts_telco_csv_and_bounds_numeric_values():
    frame = _incoming_telco_frame()

    validated = IngestionSchema.validate(frame)

    assert validated.shape == (2, 21)


def test_ingestion_schema_rejects_out_of_range_numeric_value():
    frame = _incoming_telco_frame()
    frame.loc[0, "MonthlyCharges"] = 501.0

    with pytest.raises(SchemaError):
        IngestionSchema.validate(frame)


def test_model_input_schema_rejects_customer_identifier():
    incoming = IngestionSchema.validate(_incoming_telco_frame())
    features = _normalize_columns(incoming).drop(columns=["customerID", "Churn"])

    with pytest.raises(SchemaErrors):
        ModelInputSchema.validate(features.assign(customerID=incoming["customerID"]))
