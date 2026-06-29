import joblib
from ollama_explainer import explain_prediction

model = joblib.load("outputs/tfidf_lr.joblib")

while True:
    text = input("\nEnter text (or type exit): ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])[0]

    print("Prediction:", prediction)

    explanation = explain_prediction(text, prediction)

    print("\nExplanation:")
    print(explanation)
