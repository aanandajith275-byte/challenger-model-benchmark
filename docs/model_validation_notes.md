# Model Validation Notes

## Validation setup

- 80/20 stratified train/test split
- Random state: 42
- XGBoost tuning with GridSearchCV
- 5-fold StratifiedKFold
- ROC-AUC used during tuning
- Final metrics measured on the held-out test set

## Leakage experiment

`dpd_30_plus` is a synthetic post-approval variable. It is intentionally correlated with the target to show how future information can inflate model performance.

The original German Credit dataset does not contain real feature-creation timestamps or a production repayment-history field for this experiment.

Recorded leaked Logistic Regression ROC-AUC: **0.981**.

## Current test results

| Model | ROC-AUC | Gini | KS | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 | 0.619 | 0.581 | 0.681 | 0.533 | 0.598 |
| XGBoost Challenger | 0.791 | 0.583 | 0.460 | 0.705 | 0.517 | 0.596 |

## XGBoost tuning

Best recorded parameters:

- learning_rate: 0.05
- max_depth: 3
- n_estimators: 200

Best recorded 5-fold CV ROC-AUC: **0.794**.

The test set is not used during GridSearch.

## Notes

The values above correspond to the current script, data and random seed. If those change, regenerate the outputs and update this file.
