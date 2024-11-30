import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

data = pd.read_csv('Reviews.csv')

data = data[data['Score'] != 3]
data['sentiment'] = data['Score'].map({1: 0, 2: 0, 4: 1, 5: 1})
data = data[['Text', 'sentiment']]

X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

start_time = time.time()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
vectorization_time = time.time() - start_time
print(f"Vectorization time: {vectorization_time:.2f} seconds")

model = LogisticRegression(max_iter=1000)

start_time = time.time()
model.fit(X_train_tfidf, y_train)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
