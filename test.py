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
        
    X = vectorizer.transform([name])
    prediction = model.predict(X)[0]
    print(f"Prediction: {prediction}\n")