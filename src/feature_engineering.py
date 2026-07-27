import re
import pandas as pd

# ==========================
# Load Dataset
# ==========================

DATA_PATH = "data/processed/processed_email.csv"
OUTPUT_PATH = "data/processed/featured_email.csv"

df = pd.read_csv(DATA_PATH)

# Fill missing text
df["Clean_Text"] = df["Clean_Text"].fillna("").astype(str)
df["Email Text"] = df["Email Text"].fillna("").astype(str)

# ==========================
# Suspicious Keywords
# ==========================

SUSPICIOUS_WORDS = {
    "urgent","verify","login","password","bank","account",
    "security","confirm","click","limited","offer",
    "winner","prize","gift","free","payment",
    "invoice","update","alert","reset",
    "bitcoin","crypto","wallet"
}

# ==========================
# Feature Functions
# ==========================

def count_urls(text):
    return len(re.findall(r"http[s]?://|www\\.", text))

def count_emails(text):
    return len(re.findall(r"\\S+@\\S+", text))

def count_digits(text):
    return sum(c.isdigit() for c in text)

def count_uppercase(text):
    return sum(c.isupper() for c in text)

def count_special(text):
    return len(re.findall(r"[^A-Za-z0-9\\s]", text))

def suspicious_count(text):
    words = text.lower().split()
    return sum(word in SUSPICIOUS_WORDS for word in words)

# ==========================
# Generate Features
# ==========================

df["Email_Length"] = df["Email Text"].str.len()

df["Word_Count"] = df["Clean_Text"].str.split().str.len()

df["URL_Count"] = df["Email Text"].apply(count_urls)

df["Email_Count"] = df["Email Text"].apply(count_emails)

df["Digit_Count"] = df["Email Text"].apply(count_digits)

df["Uppercase_Count"] = df["Email Text"].apply(count_uppercase)

df["Exclamation_Count"] = df["Email Text"].str.count("!")

df["Question_Count"] = df["Email Text"].str.count(r"\?")

df["Special_Char_Count"] = df["Email Text"].apply(count_special)

df["Suspicious_Word_Count"] = df["Clean_Text"].apply(suspicious_count)

# ==========================
# Save Dataset
# ==========================

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("Feature Engineering Completed")
print("=" * 60)

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSaved to:", OUTPUT_PATH)