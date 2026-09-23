# Challenger Model Benchmark

A credit-risk experiment comparing a Logistic Regression baseline with an XGBoost challenger. The project also includes a small data-leakage experiment to show how future information can inflate model performance.

## Main questions

- Does the XGBoost challenger improve on Logistic Regression?
- What happens when a post-approval variable is included?
- How do the models compare on unseen data?

## Dataset

The project uses OpenML's `credit-g` German Credit dataset.

- 1,000 observations
- 20 original predictors
- `bad = 1`
- `good = 0`

Categorical variables are one-hot encoded. The dataset is downloaded when the script runs.

## Leakage experiment

The original dataset does not contain a real 30+ days-past-due field. For the experiment, the script creates a synthetic `dpd_30_plus` variable that is strongly related to the target.

The idea is:

```text
Credit application → decision → repayment history
                                  ↑
                           future information
```

A model using repayment information to make the earlier credit decision would have access to information unavailable at decision time.

Recorded leakage experiment:

| Model | ROC-AUC |
|---|---:|
| Logistic Regression with simulated leakage | 0.981 |
| Clean Logistic Regression | 0.809 |

The simulated feature is removed before the final model comparison.

## Models

### Logistic Regression

Used as the baseline because it is simple and interpretable.

### XGBoost

Used as the challenger because boosted trees can capture nonlinear relationships and feature interactions.

XGBoost is tuned with a small `GridSearchCV` using 5-fold `StratifiedKFold` and ROC-AUC. The test set is kept separate from the grid search.

Best recorded parameters:

```text
learning_rate = 0.05
max_depth = 3
n_estimators = 200
```

Best cross-validation ROC-AUC: **0.794**

## Test results

Current recorded test-set results:

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.809 | 0.791 |
| Gini | 0.619 | 0.583 |
| KS | 0.581 | 0.460 |
| Precision | 0.681 | 0.705 |
| Recall | 0.533 | 0.517 |
| F1 | 0.598 | 0.596 |

These numbers describe the current held-out split. They are not claims about how either model would perform on another dataset.

Precision, recall and F1 use a 0.5 classification threshold. ROC-AUC and KS use predicted probabilities.

## Plots

The script generates:

- class distribution
- numerical distributions
- correlation heatmaps
- selected feature relationships
- leakage comparison
- model comparison
- ROC curve
- precision-recall curve
- confusion matrices
- SHAP summary plot

SHAP is used to inspect feature contributions for the XGBoost model.

## Project structure

```text
challenger-model-benchmark/
├── README.md
├── requirements.txt
├── src/
│   └── challenger_model.py
├── outputs/
├── docs/
│   └── model_validation_notes.md
└── data/
    └── README.md
```

## Limitations

- Small public dataset
- Synthetic leakage feature
- No real feature timestamps
- No external validation set
- Small hyperparameter grid
- 0.5 threshold is only a demonstration
- No production deployment or monitoring
- No regulatory approval or adverse-action code generation

## Run

```bash
pip install -r requirements.txt
python src/challenger_model.py
```

## Tools

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, SciPy, Matplotlib and Seaborn.

## Author

Aanand Ajith — B.Tech Mechanical Engineering, IIT Hyderabad
