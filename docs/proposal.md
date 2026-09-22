## Customer Churn Prediction

## 1. Dataset

- Name and description: IBM Telco Customer Churn dataset. It contains customer demographics, account information, and service subscriptions, with the goal of predicting customer retention.
- Source URL and license: Kaggle (IBM provided), openly available for educational use. https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
- Rough size: 7,043 rows and 21 features.

### Mini data dictionary

| Feature | Type | Description |
| --- | --- | --- |
| `tenure` | Numerical | Number of months the customer has stayed with the company. Range: 0-72. |
| `MonthlyCharges` | Numerical (USD) | Monthly fee charged to the customer. Range: $18.25 - $118.75. |
| `TotalCharges` | Numerical / string-like | Total amount charged over time. It contains hidden blank spaces for new customers with 0 months of tenure, which makes it a data-quality issue. |
| `Contract` | Categorical | Customer contract type: Month-to-month, One year, Two year. |

## 2. Prediction task

- Target column: `Churn` with values `Yes` or `No`.
- Task type: Binary classification.
- Why it is useful: Customer acquisition is much more expensive than retention. Predicting which customers are at risk of churn allows a company to offer proactive incentives and reduce revenue loss.

## 3. Suitability check

### Data and model feasibility

- The dataset is a single CSV file under 1 MB.
- It fits comfortably in memory.
- A basic scikit-learn model can train on it in well under a second.

### Known limitations

- Class imbalance: The target variable is imbalanced (about 73% `No` and 27% `Yes`), so raw accuracy can be misleading.
- Hidden missing values: The `TotalCharges` column contains empty strings instead of standard `NaN` values, so explicit preprocessing is required.

## 4. Baseline idea

- Initial model: `LogisticRegression`.
- Preprocessing: one-hot encode categorical variables and impute missing values in `TotalCharges`.
- Primary metric to optimize: `F1-score` for the positive class (`Churn = Yes`) or `ROC-AUC`.

> Because of the class imbalance, maximizing simple accuracy would lead the model to predict mostly `No`, which is not useful for identifying churners.
