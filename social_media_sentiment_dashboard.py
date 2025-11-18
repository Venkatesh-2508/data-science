import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# ---------------- STEP 1: Load dataset safely ----------------
df = pd.read_csv("comments.csv", encoding="latin1")

# Auto-fix column names (no KeyError)
df.columns = df.columns.str.strip().str.title()

# ---------------- STEP 2: Clean text ----------------
df["Clean_Comment"] = (
    df["Comment"]
    .astype(str)
    .str.lower()
    .str.replace(r'[^a-z\s]', '', regex=True)
)

# ---------------- STEP 3: Sentiment function ----------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Sentiment label
df["Sentiment"] = df["Clean_Comment"].apply(get_sentiment)

# EXTRA → Sentiment Score (numeric polarity)
df["Sentiment_Score"] = df["Clean_Comment"].apply(lambda x: TextBlob(x).sentiment.polarity)

# ---------------- STEP 4: Sentiment Counts ----------------
sentiment_counts = df["Sentiment"].value_counts()

print("\n=== Overall Sentiment Counts ===")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment} : {count}")

# EXTRA → Sentiment Percentages
total = len(df)
positive = (df["Sentiment"] == "Positive").sum()
negative = (df["Sentiment"] == "Negative").sum()
neutral  = (df["Sentiment"] == "Neutral").sum()

print("\n=== Sentiment Percentage ===")
print(f"Positive : {positive/total*100:.2f}%")
print(f"Negative : {negative/total*100:.2f}%")
print(f"Neutral  : {neutral/total*100:.2f}%")

# ---------------- STEP 5: Bar chart (premium style) ----------------
plt.figure(figsize=(7,5))
plt.bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color=["#228B22", "#B22222", "#708090"],
    edgecolor="black",
    linewidth=1.3
)
plt.title("Overall Sentiment Count", fontsize=15, fontweight='bold')
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Number Of Comments", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# ---------------- STEP 6: Platform-wise sentiment ----------------
platform_group = df.groupby(["Platform","Sentiment"]).size().unstack(fill_value=0)

print("\n=== Platform-wise Sentiment Table ===")
print(platform_group.to_string())   # clean table

platform_group.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6),
    color=["#90EE90", "#FA8072", "#D3D3D3"],
    edgecolor="black"
)
plt.title("Platform-wise Sentiment Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Platform", fontsize=12)
plt.ylabel("Number Of Comments", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------- STEP 7: Final Summary ----------------
print("\n=== Final Summary ===")
print(f"Total Comments Analyzed : {total}")
print(f"Positive Comments       : {positive}")
print(f"Negative Comments       : {negative}")
print(f"Neutral Comments        : {neutral}")

print("\nAnalysis Completed Successfully! ✅")
