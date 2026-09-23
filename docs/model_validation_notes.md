# Model Validation Notes

## Validation setup

- Train/test split: 80/20
- Stratified split
- Random state: 42
- XGBoost tuning: GridSearchCV
- Cross-validation: 5-fold StratifiedKFold
- GridSearch scoring: ROC-AUC
- Final evaluation: held-out test set

## Leakage experiment

`dpd_30_plus` is simulated as post-approval information. It is intentionally correlated with the target to demonstrate how future information can inflate model performance.

The original German Credit dataset does not contain real feature-creation timestamps or a production repayment-history field for this experiment.

## Recorded results

| Model | ROC-AUC | Gini | KS | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.812 | 0.625 | 0.532 | 0.671 | 0.522 | 0.588 |
| XGBoost Challenger | 0.782 | 0.563 | 0.432 | 0.627 | 0.356 | 0.454 |

Recorded leaked Logistic Regression ROC-AUC: **0.981**.

Best recorded XGBoost parameters:

- learning_rate: 0.05
- max_depth: 3
- n_estimators: 200

Best recorded CV ROC-AUC: **0.794**.

These values are documented from the project run and should be updated if the experiment is intentionally rerun with changed code, random seeds, or data.
