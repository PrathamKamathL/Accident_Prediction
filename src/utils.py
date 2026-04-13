import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/fsml_data.csv")

def create_encoders():
    encoders = {}

    for col in df.columns:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("Encoders saved successfully")

def create_defaults():
    defaults = {}

    for col in df.columns:
        defaults[col] = df[col].mode()[0]

    with open("models/defaults.pkl", "wb") as f:
        pickle.dump(defaults, f)