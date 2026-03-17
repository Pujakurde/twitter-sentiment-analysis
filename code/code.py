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

# ------------------------------
# Step 9: Top 20 Words Count 
# ------------------------------

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
plt.figure(figsize=(10,5))
sns.barplot(x=counts, y=words)

plt.title("Top 20 Most Common Words in Tweets")
plt.xlabel("Frequency")
plt.ylabel("Words")

# Save the figure (IMPORTANT for GitHub)
plt.savefig("D:\\Folder D\\DA\\Tweet_Sentiment_Analysis\\visuals\\top_words.png")

# Show the plot
plt.show()