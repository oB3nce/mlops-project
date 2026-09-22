# Data Dictionary — Telco Customer Churn

This document describes the dataset used for the churn prediction project. It is meant to capture the domain meaning of each field, highlight expected ranges, and document the data-quality issues that matter for model training and evaluation.

## What the dataset is

The dataset comes from the IBM Telco Customer Churn dataset, commonly shared on Kaggle and used in customer retention analytics.

- Dataset: IBM Telco Customer Churn
- Source: Kaggle / IBM public dataset for educational use
- Rows: 7,043
- Features: 20 customer-level features + 1 identifier column
- Target: `Churn` (`Yes` / `No`)
- Task: Binary classification
- Class balance: approximately 73% `No` and 27% `Yes` (class imbalance is present)
- File: `data/Telco-Customer-Churn.csv`

This is a customer retention dataset: the goal is to predict which customers are likely to leave the telecom service so that retention actions can be targeted at the right people.

## Feature dictionary

| Column | Meaning | Unit / type | Plausible range | Missing/sentinel pattern | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customerID` | Unique customer identifier | String | N/A | No | This is an ID column, not a predictive feature. It should usually be dropped before modeling. |
| `gender` | Customer gender | Categorical | Female / Male | No | A demographic variable; not necessarily a causal driver of churn. |
| `SeniorCitizen` | Whether the customer is 65 or older | Binary (0/1) | 0 or 1 | No | Encoded as numeric, but conceptually binary. |
| `Partner` | Whether the customer has a partner | Categorical | Yes / No | No | Relationship status indicator. |
| `Dependents` | Whether the customer has dependents | Categorical | Yes / No | No | Household dependency indicator. |
| `tenure` | Number of months the customer stayed with the company | Integer (months) | 0–72 | No | Lower values may indicate recent customers and higher churn risk. |
| `PhoneService` | Whether the customer has phone service | Categorical | Yes / No | No | A service-level flag. |
| `MultipleLines` | Whether the customer has multiple phone lines | Categorical | Yes / No / No phone service | No | Distinct from `No` because “No phone service” is a meaningful category. |
| `InternetService` | Internet provider type | Categorical | DSL / Fiber optic / No | No | Important segmentation feature. |
| `OnlineSecurity` | Whether the customer has online security service | Categorical | Yes / No / No internet service | No | Can be a proxy for service engagement. |
| `OnlineBackup` | Whether the customer has online backup service | Categorical | Yes / No / No internet service | No | Service subscription feature. |
| `DeviceProtection` | Whether the customer has device protection | Categorical | Yes / No / No internet service | No | Another add-on service variable. |
| `TechSupport` | Whether the customer has technical support | Categorical | Yes / No / No internet service | No | Service quality indicator. |
| `StreamingTV` | Whether the customer uses streaming TV service | Categorical | Yes / No / No internet service | No | Indicates bundled entertainment usage. |
| `StreamingMovies` | Whether the customer uses streaming movies service | Categorical | Yes / No / No internet service | No | Indicates bundled entertainment usage. |
| `Contract` | Customer contract length | Categorical | Month-to-month / One year / Two year | No | Strong churn predictor. |
| `PaperlessBilling` | Whether the customer receives bills electronically | Categorical | Yes / No | No | Billing behavior indicator. |
| `PaymentMethod` | Payment method used by the customer | Categorical | Electronic check / Mailed check / Bank transfer / Credit card | No | Payment style often correlates with churn behavior. |
| `MonthlyCharges` | Monthly telecom charges | Numeric (USD) | ~$18.25–$118.75 | No | Monthly billing amount. |
| `TotalCharges` | Total lifetime charges up to the current date | Numeric (USD) | Usually 0–up to several thousand | Potential hidden missing values / blank strings | This column is known to have formatting issues; zero-length or whitespace values may appear for new or incomplete accounts. |
| `Churn` | Whether the customer left the company | Categorical / target | Yes / No | No | Raw file uses text labels; for modeling, this is usually converted to 0/1. |

## Domain-expert notes (read these — they explain the gotchas)

- The target is imbalanced. Roughly 27% of customers churned, so accuracy alone is not a good evaluation metric. F1-score, precision/recall, or ROC-AUC are typically more informative.
- `customerID` is unique for each row and should not be used as a model feature. It is useful for tracking records, not for prediction.
- `TotalCharges` is the most important data-quality issue in this dataset. It is stored in a way that can include empty or whitespace-like values, especially for customers with low or zero tenure, and may need conversion to numeric type during preprocessing.
- Some variables encode categories with a meaningful third option such as `No internet service` rather than a simple binary `Yes/No`. This should not always be collapsed into plain `No` without considering the domain meaning.
- `SeniorCitizen` is numeric-looking, but it is really a binary flag. Depending on the modeling pipeline, it may be treated as categorical or numeric.
- `Contract` and `tenure` are likely to be among the strongest churn predictors. Longer contracts and longer customer history are often associated with lower churn risk.
- There is no obvious temporal dimension in the raw CSV; this is a static snapshot of customer account states rather than an event stream.

## Known limitations and caveats

- The dataset is observational and not a controlled experiment. It should be used for practical churn prediction, not causal conclusion about why customers churn.
- The file contains a mix of categorical and numeric variables, so the preprocessing pipeline must handle one-hot encoding or equivalent encoding for tree-based models or linear models.
- This is a public demo dataset, so it is suitable for learning and experimentation, but it does not capture all business realities of a live telecom company.
- If the target is converted from `Yes`/`No` to `1`/`0`, the conversion step should be done consistently across train/test splits and evaluation metrics.

## Source & citation

- IBM Telco Customer Churn dataset, widely circulated via Kaggle for educational and applied ML use.
- Intended use: customer churn forecasting, customer retention analysis, and model evaluation in a business context.
