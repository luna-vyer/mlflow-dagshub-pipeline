from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# Charger le modèle à partir du dossier mlruns
model = mlflow.pyfunc.load_model("mlruns/0/e3e7ceb17bf24266aef4f17259af4208/artifacts/random_forest_model")

@app.get("/")
def index():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([list(data.values())])
    return {"prediction": prediction.tolist()}
