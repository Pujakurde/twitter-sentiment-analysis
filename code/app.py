from fastapi import FastAPI
import joblib
import re
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize app
app = FastAPI()

# Stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower()

def remove_stopwords(text):
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

# Home route
@app.get("/")
def home():
    return {"message": "Twitter Sentiment API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(text: str):
    cleaned = remove_stopwords(clean_tweet(text))
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    return {"sentiment": prediction}