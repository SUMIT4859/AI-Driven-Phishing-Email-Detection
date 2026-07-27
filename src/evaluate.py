import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

warnings.filterwarnings("ignore")

# ======================================================
# Create Result Folders
# ======================================================

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("data/processed/featured_email.csv")

df["Clean_Text"] = df["Clean_Text"].fillna("").astype(str)

numeric_columns = [
    "Email_Length",
    "Word_Count",
    "URL_Count",
    "Email_Count",
    "Digit_Count",
    "Uppercase_Count",
    "Exclamation_Count",
    "Question_Count",
    "Special_Char_Count",
    "Suspicious_Word_Count",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df[df["Clean_Text"].str.strip() != ""]

X = df[["Clean_Text"] + numeric_columns]
y = df["Label"]

# ======================================================
# Same Train/Test Split
# ======================================================

from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ======================================================
# Load Best Model
# ======================================================

model = joblib.load("models/best_model.pkl")

pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    prob = model.predict_proba(X_test)[:, 1]
else:
    prob = None

# ======================================================
# Metrics
# ======================================================

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics.to_csv(
    "results/metrics/evaluation_metrics.csv",
    index=False
)

print(metrics)

# ======================================================
# Classification Report
# ======================================================

report = classification_report(y_test, pred)

with open(
    "results/reports/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ======================================================
# Confusion Matrix
# ======================================================

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "results/figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ======================================================
# ROC Curve
# ======================================================

if prob is not None:

    RocCurveDisplay.from_predictions(
        y_test,
        prob
    )

    plt.title("ROC Curve")

    plt.savefig(
        "results/figures/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ======================================================
# Precision Recall Curve
# ======================================================

if prob is not None:

    PrecisionRecallDisplay.from_predictions(
        y_test,
        prob
    )

    plt.title("Precision Recall Curve")

    plt.savefig(
        "results/figures/precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ======================================================
# Feature Importance
# ======================================================

try:

    clf = model.named_steps["classifier"]

    pre = model.named_steps["preprocessor"]

    tfidf = pre.named_transformers_["text"]

    feature_names = list(tfidf.get_feature_names_out())

    feature_names.extend(numeric_columns)

    if hasattr(clf, "feature_importances_"):

        importance = clf.feature_importances_

        top = np.argsort(importance)[-20:]

        plt.figure(figsize=(10,6))

        plt.barh(
            np.array(feature_names)[top],
            importance[top]
        )

        plt.title("Top 20 Feature Importance")

        plt.savefig(
            "results/figures/feature_importance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

except Exception:
    print("Feature importance skipped.")

print("\nEvaluation Completed Successfully!")