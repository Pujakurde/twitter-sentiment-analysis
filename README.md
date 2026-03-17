# Twitter Sentiment Analysis using NLP

## 📌 Project Overview

This project analyzes Twitter data to understand public sentiment using **Natural Language Processing (NLP)** techniques.

The analysis explores patterns in tweets and visualizes sentiment trends using Python.

---

## 🛠 Tools & Technologies

* Python
* Pandas
* NLTK
* Matplotlib
* Seaborn
* WordCloud

---

## 📂 Dataset

Dataset used: **Sentiment140 Twitter Dataset**

Source:
https://www.kaggle.com/datasets/kazanova/sentiment140

The dataset contains **1.6 million tweets labeled with sentiment**.

Sentiment labels:

* **0 → Negative**
* **4 → Positive**

⚠️ Note:
The dataset is **not included in this repository** due to GitHub's file size limit.
Download the dataset from Kaggle and place it inside a `dataset/` folder.

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading

* Load Twitter dataset using Pandas
* Select relevant columns

### 2️⃣ Data Cleaning

* Remove URLs
* Remove mentions (@username)
* Remove hashtags
* Remove punctuation
* Convert text to lowercase

### 3️⃣ Stopword Removal

* Remove common words such as:
  *the, is, and, to*

### 4️⃣ Text Processing

* Create cleaned tweet text for analysis

### 5️⃣ Data Visualization

* Word Cloud of most frequent words
* Sentiment distribution chart

### 6️⃣ Word Frequency Analysis

* Extract top 20 most common words
* Visualize using bar chart

---

## 📊 Visualizations

### Word Cloud

Shows the most common words appearing in tweets.

![WordCloud](visuals/wordcloud.png)

### Sentiment Distribution

Displays the number of positive and negative tweets.

![Sentiment Chart](visuals/sentiment_chart.png)

### Top 20 Most Common Words

This visualization shows the most frequently occurring words in tweets after preprocessing and stopword removal.

![Top Words](visuals/top_words.png)
---

## 📁 Project Structure

```
twitter-sentiment-analysis
│
├── code
│   └── code.py
│
├── visuals
│   ├── wordcloud.png
│   ├── sentiment_chart.png
│   └── top_words.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run the script:

```
python code/code.py
```

---

## 📈 Insights

* Common words in tweets include **love, good, day, happy, and miss**
* Tweets reflect strong emotional expressions and opinions
* Both positive and negative sentiments are well represented in the dataset
* Word frequency analysis helps identify common patterns in user behavior

---

## 👩‍💻 Author

**Puja Kurde**
Data Science Student

GitHub: https://github.com/Pujakurde
