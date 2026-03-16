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
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#\w+", "", text)         # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove punctuation
    text = text.lower()                     # convert to lowercase
    return text


# Apply cleaning function
df["clean_text"] = df["text"].apply(clean_tweet)


# ------------------------------
# Step 6: Remove stopwords
# ------------------------------
# Stopwords are common words like:
# the, is, and, to, etc.

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


# Apply stopword removal
df["clean_text"] = df["clean_text"].apply(remove_stopwords)


# ------------------------------
# Step 7: Create WordCloud
# ------------------------------
# WordCloud shows most common words
# appearing in tweets.

text_data = " ".join(df["clean_text"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text_data)


plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Tweets")
plt.show()


# ------------------------------
# Step 8: Sentiment Distribution
# ------------------------------
# Visualize how many tweets are positive
# and negative.

sns.countplot(x="sentiment", data=df)

plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.show()