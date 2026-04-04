import pickle
import pandas as pd

with open("models/model_v2.pkl", "rb") as f:
    model = pickle.load(f)

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    
    prediction = model.predict(df)
    return int(prediction[0])