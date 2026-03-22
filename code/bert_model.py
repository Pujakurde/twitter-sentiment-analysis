from transformers import pipeline

# Load pre-trained sentiment model
classifier = pipeline("sentiment-analysis")

# Test predictions
texts = [
    "I love this product",
    "This is the worst experience ever"
]

for text in texts:
    result = classifier(text)
    print(f"Text: {text}")
    print("Prediction:", result)