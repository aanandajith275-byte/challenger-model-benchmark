import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid")
np.random.seed(42)

data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
df = data.frame.copy()

df["target"] = (df["class"] == "bad").astype(int)
df.drop(columns="class", inplace=True)

print("Dataset:", df.shape)

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="target")
plt.xticks([0, 1], ["Good", "Bad"])
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "class_distribution.png"), dpi=200)
plt.close()

num = df.select_dtypes(include=np.number).columns.drop("target")

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a, col in zip(ax.flat, num):
    sns.histplot(df[col], kde=True, ax=a)
    a.set_title(col)
for a in ax.flat[len(num):]:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "numerical_distributions.png"), dpi=200)
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[list(num) + ["target"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Numerical Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "numerical_correlation.png"), dpi=200)
plt.close()

pairs = [
    ("duration", "credit_amount"),
    ("age", "credit_amount"),
    ("age", "duration"),
    ("credit_amount", "duration")
]

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a, (x, y) in zip(ax.flat, pairs):
    sns.scatterplot(data=df, x=x, y=y, hue="target", alpha=.7, ax=a)
    a.set_title(f"{x} vs {y}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "selected_relationships.png"), dpi=200)
plt.close()

X = pd.get_dummies(df.drop(columns="target"), drop_first=True)
y = df["target"]
X.columns = X.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)

print("Encoded features:", X.shape[1])

plt.figure(figsize=(18, 14))
sns.heatmap(X.assign(target=y).corr(), cmap="coolwarm", center=0)
plt.title("Full Encoded Feature Correlation")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "full_feature_heatmap.png"), dpi=200)
plt.close()

rng = np.random.default_rng(42)
dpd = np.where(
    y == 1,
    rng.binomial(1, .99, len(y)),
    rng.binomial(1, .02, len(y))
)

X_leaked = X.copy()
X_leaked["dpd_30_plus"] = dpd

XL_train, XL_test, y_train, y_test = train_test_split(
    X_leaked, y, test_size=.2, random_state=42, stratify=y
)

leaked_model = LogisticRegression(max_iter=3000)
leaked_model.fit(XL_train, y_train)
leaked_prob = leaked_model.predict_proba(XL_test)[:, 1]
leaked_auc = roc_auc_score(y_test, leaked_prob)

print("Leaked Logistic AUC:", round(leaked_auc, 3))

X = X_leaked.drop(columns="dpd_30_plus")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)

log_model = LogisticRegression(max_iter=3000)
log_model.fit(X_train, y_train)
log_prob = log_model.predict_proba(X_test)[:, 1]
log_pred = (log_prob >= .5).astype(int)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

params = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [.03, .05]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    xgb, params, scoring="roc_auc", cv=cv, n_jobs=-1
)
grid.fit(X_train, y_train)

xgb_model = grid.best_estimator_
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_prob >= .5).astype(int)

print("Best XGBoost:", grid.best_params_)
print("Best CV AUC:", round(grid.best_score_, 3))

def metrics(name, prob, pred):
    auc = roc_auc_score(y_test, prob)
    return {
        "Model": name,
        "ROC-AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_2samp(prob[y_test == 1], prob[y_test == 0]).statistic,
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    }

results = pd.DataFrame([
    metrics("Logistic Regression", log_prob, log_pred),
    metrics("XGBoost Challenger", xgb_prob, xgb_pred)
])

print("\nMODEL RESULTS")
print(results.round(3))

results.to_csv(os.path.join(OUT, "model_results.csv"), index=False)
pd.DataFrame(grid.cv_results_).to_csv(
    os.path.join(OUT, "gridsearch_results.csv"), index=False
)

results.set_index("Model").plot(kind="bar", figsize=(11, 6))
plt.ylim(0, 1)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 6))
for prob, name in [(log_prob, "Logistic Regression"), (xgb_prob, "XGBoost")]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 6))
for prob, name in [(log_prob, "Logistic Regression"), (xgb_prob, "XGBoost")]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "precision_recall_curve.png"), dpi=200)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, pred, name in zip(
    ax, [log_pred, xgb_pred], ["Logistic Regression", "XGBoost"]
):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d",
                cmap="Blues", ax=a)
    a.set_title(name)
    a.set_xlabel("Predicted")
    a.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=200)
plt.close()

clean_auc = results.loc[
    results["Model"] == "Logistic Regression", "ROC-AUC"
].iloc[0]

leak = pd.DataFrame({
    "Model": ["With Leakage", "Without Leakage"],
    "ROC-AUC": [leaked_auc, clean_auc]
})

plt.figure(figsize=(7, 5))
sns.barplot(data=leak, x="Model", y="ROC-AUC")
plt.ylim(0, 1)
plt.title("Effect of Simulated Data Leakage")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "leakage_audit.png"), dpi=200)
plt.close()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "shap_importance.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nAll outputs saved to:")
print(os.path.abspath(OUT))
