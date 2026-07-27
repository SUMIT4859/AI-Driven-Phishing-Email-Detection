import os
import joblib
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ==========================================
# Create Folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

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
    "Suspicious_Word_Count"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df[df["Clean_Text"].str.strip() != ""]

X = df[["Clean_Text"] + numeric_columns]
y = df["Label"]

# ==========================================
# Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Pipeline
# ==========================================

text_transformer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("text", text_transformer, "Clean_Text"),
    ("num", numeric_transformer, numeric_columns)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

# ==========================================
# Parameter Grid
# ==========================================

param_grid = {
    "classifier__C": [0.1, 1, 5, 10],
    "classifier__solver": ["liblinear", "lbfgs"],
    "classifier__max_iter": [500, 1000]
}

print("=" * 60)
print("Running GridSearchCV...")
print("=" * 60)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# ==========================================
# Best Model
# ==========================================

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("\n")
print("=" * 60)
print("Best Parameters")
print("=" * 60)

print(grid.best_params_)

print("\n")
print("=" * 60)
print("Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, pred))

joblib.dump(best_model, "models/tuned_logistic_regression.pkl")

print("\nSaved: models/tuned_logistic_regression.pkl")