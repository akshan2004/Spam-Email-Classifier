from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        message = request.form["message"]

        # Convert text into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Prediction
        result = model.predict(message_tfidf)[0]

        # Prediction probability
        probabilities = model.predict_proba(message_tfidf)[0]
        confidence = round(max(probabilities) * 100, 2)

        if result == 1:
            prediction = "SPAM"
        else:
            prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)