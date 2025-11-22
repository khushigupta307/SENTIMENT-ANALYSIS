# Sentiment Analysis of Social Media Posts using Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample social media posts (you can replace with dataset)
posts = [
    "I love this product! It's amazing.",
    "Worst experience ever, I hate it.",
    "Feeling happy today!",
    "This is so bad, very disappointed.",
    "Great service, I am satisfied.",
    "Terrible quality, not worth it."
]

# Corresponding labels (1 = Positive, 0 = Negative)
labels = [1, 0, 1, 0, 1, 0]

# Step 1: Convert text to numerical vectors
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(posts)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Train ML model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict
pred = model.predict(X_test)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Test on a new social media post
new_post = ["I really enjoyed using this app!"]
new_vec = tfidf.transform(new_post)
prediction = model.predict(new_vec)

if prediction[0] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
