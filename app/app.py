from fastapi import FastAPI, HTTPException
from app.schema import AccidentInput
from src.predict import predict
import pickle

with open("models/defaults.pkl", "rb") as f:
    defaults = pickle.load(f)

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
        for key, value in input_dict.items():
            if key == "Number_of_casualties" and (value is None or value==0):
                input_dict[key] = 1
            if isinstance(value, str) and value.strip() == "":
                input_dict[key] = defaults.get(key)
        result = predict(input_dict)
        prediction = result["class_id"]
        confidence = result["confidence"]
        predicted_label = severity_mapping.get(prediction, "Unknown")

        return {
            "prediction": predicted_label,
            "class_id": prediction,
            "confidence": round(confidence, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))