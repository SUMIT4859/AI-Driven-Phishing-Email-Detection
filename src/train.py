import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ======================================================
# Create folders
# ======================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

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
    "Suspicious_Word_Count"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df[df["Clean_Text"].str.strip() != ""]

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)
print(df.shape)

X = df[["Clean_Text"] + numeric_columns]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================================
# Models
# ======================================================

models = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "numeric_scaling": True,
        "text_only": False
    },

    "Naive Bayes": {
        "model": MultinomialNB(),
        "numeric_scaling": False,
        "text_only": True
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "numeric_scaling": False,
        "text_only": False
    },

    "Neural Network": {
        "model": MLPClassifier(
            hidden_layer_sizes=(128,64),
            max_iter=300,
            random_state=42
        ),
        "numeric_scaling": True,
        "text_only": False
    }

}

results = []

best_accuracy = 0
best_model = None
best_name = ""

# ======================================================
# Training
# ======================================================

for name, config in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if config["text_only"]:

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=5000,
                        stop_words="english"
                    ),
                    "Clean_Text"
                )
            ]
        )

    else:

        if config["numeric_scaling"]:

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

        else:

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])

        preprocessor = ColumnTransformer(
            transformers=[

                (
                    "text",
                    TfidfVectorizer(
                        max_features=5000,
                        stop_words="english"
                    ),
                    "Clean_Text"
                ),

                (
                    "num",
                    numeric_pipeline,
                    numeric_columns
                )
            ]
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", config["model"])
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    filename = (
        "models/" +
        name.lower().replace(" ", "_") +
        ".pkl"
    )

    joblib.dump(pipeline, filename)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_name = name

# ======================================================
# Save Best Model
# ======================================================

joblib.dump(best_model, "models/best_model.pkl")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(results_df)

print("\nBest Model :", best_name)
print("Best Accuracy :", round(best_accuracy,4))

print("\nSaved Models")

for file in os.listdir("models"):
    print("✔", file)

print("\nTraining Completed Successfully!")