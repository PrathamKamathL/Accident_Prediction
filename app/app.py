from fastapi import FastAPI
from app.schema import AccidentInput
from src.predict import predict

app = FastAPI()

severity_mapping = {
    0: "Fatal Injury",
    1: "Serious Injury",
    2: "Slight Injury"
}

@app.get("/")
def read_root():
    return {"message": "Accident Severity Prediction API is running"}

@app.post("/predict")
def predict_accident(data: AccidentInput):
    try:
        input_dict = data.dict()
        prediction = predict(input_dict)
        predicted_label = severity_mapping.get(prediction, "Unknown")

        return {
            "prediction": predicted_label,
            "class_id": prediction   # optional but useful
        }
    except Exception as e:
        return {
            "error": str(e)
        }