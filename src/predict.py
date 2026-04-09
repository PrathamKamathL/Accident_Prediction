import pickle
import pandas as pd

with open("models/model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


def transform_input(input_data: dict):
    df = pd.DataFrame([input_data])

    for col in df.columns:
        if col in encoders:
            le = encoders[col]

            if df[col].iloc[0] not in le.classes_:
                raise ValueError(f"Invalid value '{df[col].iloc[0]}' for {col}")

            try:
                df[col] = le.transform(df[col])
            except ValueError:
                raise ValueError(f"Unknown category '{df[col].iloc[0]}' in column '{col}'")

    return df


def predict(input_data: dict):
    df = transform_input(input_data)
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    return {
        "class_id": int(prediction),
        "confidence": float(confidence)
    }