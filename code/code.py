# ==============================
# Day 1: Data Loading & Understanding
# Twitter Sentiment Analysis Project
# ==============================

# Import required libraries
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from collections import Counter


# ------------------------------
# Step 1: Define column names
# ------------------------------
# The Sentiment140 dataset does not contain column headers,
# so we manually define them.

columns = ["sentiment", "id", "date", "query", "user", "text"]


# ------------------------------
# Step 2: Load dataset
# ------------------------------
# We load only 100,000 rows to make processing faster.

df = pd.read_csv(
    r"D:\Folder D\DA\Tweet_Sentiment_Analysis\dataset\training.csv",
    encoding="latin-1",
    names=columns,
    nrows=100000
)


# ------------------------------
# Step 3: Keep only useful columns
# ------------------------------
# For sentiment analysis we only need:
# sentiment label and tweet text.

df = df[["sentiment", "text"]]

# Convert sentiment labels
df["sentiment"] = df["sentiment"].replace({0: "Negative", 4: "Positive"})

# ------------------------------
# Step 4: Check dataset information
# ------------------------------

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Counts:")
print(df["sentiment"].value_counts())


# ------------------------------
# Step 5: Clean tweet text
# ------------------------------
# Remove URLs, mentions, hashtags, punctuation
# and convert text to lowercase.
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text) # remove URLs
    text = re.sub(r"@\w+", "", text) # remove mentions
    text = re.sub(r"#\w+", "", text)# remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)# remove punctuation
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.lower().strip()


# Apply cleaning function
df["clean_text"] = df["text"].apply(clean_tweet)


# ------------------------------
# Step 6: Remove stopwords
# ------------------------------
# Stopwords are common words like:
# the, is, and, to, etc.
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Add custom stopwords (VERY IMPORTANT)
stop_words.update([
    "im", "amp", "u", "get", "go", "like",
    "one", "dont", "cant", "thats", "youre"
])

def remove_stopwords(text):
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

df["clean_text"] = df["clean_text"].apply(remove_stopwords)
# Remove empty tweets after cleaning
df = df[df["clean_text"].str.strip() != ""]

# ------------------------------
# Step 7: Create WordCloud
# ------------------------------
# WordCloud shows most common words
# appearing in tweets.

text_data = " ".join(df["clean_text"])

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="coolwarm",
    max_words=100,
    collocations=False
).generate(text_data)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Tweets")

plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\wordcloud.png")

plt.show()


# ------------------------------
# Step 8: Sentiment Distribution
# ------------------------------
# Visualize how many tweets are positive
# and negative.
plt.figure(figsize=(6,4))

sns.countplot(x="sentiment", data=df, palette="Set2")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\sentiment_chart.png")
plt.show()


# Combine all cleaned tweets into one string and split into words
all_words = " ".join(df["clean_text"]).split()

# Count frequency of each word
word_counts = Counter(all_words)

# Get top 20 most common words
common_words = word_counts.most_common(20)

# Split into two lists for plotting
words = [word[0] for word in common_words]
counts = [word[1] for word in common_words]

# Plot bar chart
plt.figure(figsize=(12,6))

sns.barplot(
    x=counts,
    y=words,
    palette="viridis"
)

plt.title("Top 20 Most Common Words", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Words")

plt.tight_layout()

# Save the figure
plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\top_words.png")

# Show the plot
plt.show()

# ------------------------------
# Step 8: Tweet Length Analysis
# ------------------------------
df["tweet_length"] = df["clean_text"].apply(lambda x: len(x.split()))
plt.figure(figsize=(12,6))

sns.histplot(
    data=df,
    x="tweet_length",
    hue="sentiment",
    bins=40,
    kde=True
)

plt.title("Tweet Length Distribution", fontsize=14)
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\tweet_length.png")
plt.show()

# ------------------------------
# Step 9: Positive vs Negative WordCloud
# ------------------------------

positive_text = " ".join(df[df["sentiment"]=="Positive"]["clean_text"])
negative_text = " ".join(df[df["sentiment"]=="Negative"]["clean_text"])

# Positive WordCloud
plt.figure(figsize=(10,5))
wordcloud_pos = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Greens",
    max_words=100,
    collocations=False
).generate(positive_text)

plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Tweets Word Cloud")

plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\positive_wordcloud.png")
plt.show()


# Negative WordCloud
plt.figure(figsize=(10,5))
wordcloud_neg = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Reds",
    max_words=100,
    collocations=False
).generate(negative_text)

plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Tweets Word Cloud")

plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\negative_wordcloud.png")
plt.show()

# ------------------------------
# Step 10: Sentiment Percentage
# ------------------------------
sentiment_percent = df["sentiment"].value_counts(normalize=True) * 100
print("\nSentiment Percentage:")
for sentiment, value in sentiment_percent.items():
    print(f"{sentiment}: {value:.2f}%")