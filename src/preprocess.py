import pandas as pd
import re
from bs4 import BeautifulSoup

# ==============================
# Load Dataset
# ==============================

DATA_PATH = "data/raw/phishing_email.csv"
OUTPUT_PATH = "data/processed/processed_email.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Original Dataset Shape")
print("=" * 60)
print(df.shape)

# ==============================
# Remove unnecessary column
# ==============================

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ==============================
# Remove missing emails
# ==============================

df.dropna(subset=["Email Text"], inplace=True)

# ==============================
# Remove duplicate emails
# ==============================

df.drop_duplicates(subset=["Email Text"], inplace=True)

# ==============================
# Convert labels
# ==============================

label_mapping = {
    "Safe Email": 0,
    "Phishing Email": 1
}

df["Label"] = df["Email Type"].map(label_mapping)

# ==============================
# Text Cleaning Function
# ==============================

def clean_email(text):

    # Remove HTML
    text = BeautifulSoup(str(text), "html.parser").get_text()

    # Replace URLs
    text = re.sub(r"http\\S+|www\\S+", " URL ", text)

    # Replace Email Addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==============================
# Apply Cleaning
# ==============================

df["Clean_Text"] = df["Email Text"].apply(clean_email)

# ==============================
# Save Dataset
# ==============================

df.to_csv(OUTPUT_PATH, index=False)

# ==============================
# Summary
# ==============================

print("\nDataset Successfully Cleaned!\n")

print("=" * 60)
print("Final Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("Class Distribution")
print("=" * 60)
print(df["Label"].value_counts())

print("\n")

print("=" * 60)
print("Sample Clean Email")
print("=" * 60)
print(df["Clean_Text"].iloc[0][:500])

print("\n")

print("Processed dataset saved to:")
print(OUTPUT_PATH)