import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===========================
# Load Processed Dataset
# ===========================

DATA_PATH = "data/processed/processed_email.csv"

df = pd.read_csv(DATA_PATH)

# ===========================
# Create Output Folder
# ===========================

os.makedirs("results/figures", exist_ok=True)

# ===========================
# Dataset Summary
# ===========================

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("Class Distribution")
print("=" * 60)
print(df["Label"].value_counts())

# ===========================
# Class Distribution Chart
# ===========================

counts = df["Label"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(["Safe", "Phishing"], counts.values)
plt.title("Email Class Distribution")
plt.xlabel("Email Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/figures/class_distribution.png")
plt.close()

print("✓ Saved: class_distribution.png")

# ===========================
# Email Length Distribution
# ===========================

df["Email_Length"] = df["Clean_Text"].str.len()

plt.figure(figsize=(8,5))
plt.hist(df["Email_Length"], bins=40)
plt.title("Email Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/figures/email_length_distribution.png")
plt.close()

print("✓ Saved: email_length_distribution.png")

# ===========================
# Word Clouds
# ===========================

safe_text = " ".join(
    df[df["Label"] == 0]["Clean_Text"].fillna("").astype(str)
)

phishing_text = " ".join(
    df[df["Label"] == 1]["Clean_Text"].fillna("").astype(str)
)

safe_cloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(safe_text)

plt.figure(figsize=(12,6))
plt.imshow(safe_cloud)
plt.axis("off")
plt.tight_layout()
plt.savefig("results/figures/safe_wordcloud.png")
plt.close()

print("✓ Saved: safe_wordcloud.png")

phishing_cloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(phishing_text)

plt.figure(figsize=(12,6))
plt.imshow(phishing_cloud)
plt.axis("off")
plt.tight_layout()
plt.savefig("results/figures/phishing_wordcloud.png")
plt.close()

print("✓ Saved: phishing_wordcloud.png")

print("\nEDA Completed Successfully!")