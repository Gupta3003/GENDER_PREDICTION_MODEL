import joblib
import os

os.makedirs("models", exist_ok=True)

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/gender_model.pkl")

print("System ready!\n")

while True:
    name = input("Enter a name to predict (or type 'exit' to stop): ").strip()

    if name.lower() == 'exit':
        break

    # Transform input
    X = vectorizer.transform([name])

    # Prediction
    prediction = model.predict(X)[0]

    # Probability
    probabilities = model.predict_proba(X)[0]

    # Classes
    classes = model.classes_

    # Create probability dictionary
    all_probabilities = {
        cls: round(float(prob), 2)
        for cls, prob in zip(classes, probabilities)
    }

    # Confidence
    max_probability = max(probabilities)

    if max_probability >= 0.85:
        confidence = "high"
    elif max_probability >= 0.65:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "input": name,
        "predicted_gender": prediction,
        "confidence": confidence,
        "probability": round(float(max_probability), 2),
        "all_probabilities": all_probabilities
    }

    print(result)
    print()

