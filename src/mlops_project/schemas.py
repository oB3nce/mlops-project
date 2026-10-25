from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class IngestionSchema(pa.DataFrameModel):
    """Contract for rows as represented in the incoming Telco CSV."""

    customerID: Series[str] = pa.Field(
        str_matches=r"^\d{4}-[A-Z]{5}$",
        unique=True,
        nullable=False,
    )
    gender: Series[str] = pa.Field(isin=["Female", "Male"], nullable=False)
    SeniorCitizen: Series[int] = pa.Field(ge=0, le=1, nullable=False)
    Partner: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    tenure: Series[int] = pa.Field(ge=0, le=120, nullable=False)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    MultipleLines: Series[str] = pa.Field(
        isin=["Yes", "No", "No phone service"],
        nullable=False,
    )
    InternetService: Series[str] = pa.Field(
        isin=["DSL", "Fiber optic", "No"],
        nullable=False,
    )
    OnlineSecurity: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    OnlineBackup: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    DeviceProtection: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    TechSupport: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    StreamingTV: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    StreamingMovies: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"],
        nullable=False,
    )
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        nullable=False,
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0.0, le=500.0, nullable=False)
    TotalCharges: Series[str] = pa.Field(
        str_matches=r"^\s*(?:\d+(?:\.\d*)?|\.\d+)?\s*$",
        nullable=False,
    )
    Churn: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)

    class Config:
        strict = True
        coerce = True


class ModelInputSchema(pa.DataFrameModel):
    """Contract for the feature columns passed into the model pipeline."""

    gender: Series[str] = pa.Field(isin=["Female", "Male"], nullable=False)
    SeniorCitizen: Series[int] = pa.Field(ge=0, le=1, nullable=False)
    Partner: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    tenure: Series[int] = pa.Field(ge=0, le=120, nullable=False)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    MultipleLines: Series[str] = pa.Field(
        isin=["Yes", "No", "No phone service"],
        nullable=False,
    )
    InternetService: Series[str] = pa.Field(
        isin=["DSL", "Fiber optic", "No"],
        nullable=False,
    )
    OnlineSecurity: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    OnlineBackup: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    DeviceProtection: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    TechSupport: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    StreamingTV: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    StreamingMovies: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"],
        nullable=False,
    )
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"],
        nullable=False,
    )
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"], nullable=False)
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        nullable=False,
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0.0, le=500.0, nullable=False)
    TotalCharges: Series[float] = pa.Field(ge=0.0, le=60_000.0, nullable=True)

    # The unique customer ID is useful for record tracking but would let a model memorize customers.
    class Config:
        strict = True
        coerce = True
