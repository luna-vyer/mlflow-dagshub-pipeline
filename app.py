from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load("random_forest_model.pkl")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Model API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
