import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 🔹 Fake dataset (username → fake/real)
# Real data would come from Twitter API later
usernames = [
    "john_doe", "mary_smith", "cool_guy", "tech_lover", "student123",
    "bot_account", "fake_user_99", "scammer007", "buy_followers", "spam_bot"
]

labels = [1, 1, 1, 1, 1,   # Real profiles = 1
          0, 0, 0, 0, 0]   # Fake profiles = 0

# Text vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(usernames)

# Logistic Regression classifier
model = LogisticRegression()
model.fit(X, labels)

# Save model + vectorizer
joblib.dump(model, "fake_profile_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved!")
