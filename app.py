from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()


model = mlflow.pyfunc.load_model("model")

@app.get("/")
def index():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([list(data.values())])
    return {"prediction": prediction.tolist()}
