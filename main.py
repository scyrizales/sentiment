import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

# Preparing data
nltk.download("movie_reviews")

documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Converting to data frame
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Model training
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split data between training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: { accuracy_score(y_test, y_pred) }")
print(f"Classification report:\n{ classification_report(y_test, y_pred) }")

# Prediction
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

print(predict_sentiment("I absolutely love this movie. It's fantastic."))
print(predict_sentiment("It was a terrible film."))
print(predict_sentiment("The movie was okay, nothing special."))
