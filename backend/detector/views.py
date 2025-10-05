import os
import joblib
from django.shortcuts import render

APP_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(APP_DIR, "fake_profile_model.pkl"))
vectorizer = joblib.load(os.path.join(APP_DIR, "vectorizer.pkl"))

def index(request):
    result = None
    if request.method == "POST":
        username = request.POST.get("username")
        if username:
            X_test = vectorizer.transform([username])
            prediction = model.predict(X_test)[0]
            result = f"Username: {username} → {'Real Profile ✅' if prediction == 1 else 'Fake Profile ❌'}"
        else:
            result = "Please enter a username!"

    return render(request, "detector/index.html", {"result": result})
