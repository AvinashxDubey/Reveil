import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 🔹 Dummy dataset (username → fake/real)
# 1 = Real, 0 = Fake
usernames = [
    "john_doe", "mary_smith", "cool_guy", "tech_lover", "student123",
    "bot_account", "fake_user_99", "scammer007", "buy_followers", "spam_bot"
]

labels = [1, 1, 1, 1, 1,   # Real profiles
          0, 0, 0, 0, 0]   # Fake profiles

# Vectorize usernames
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(usernames)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, labels)

# Save model + vectorizer in detector folder
joblib.dump(model, "fake_profile_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved as .pkl files!")
