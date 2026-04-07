import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/fsml_data.csv")

encoders = {}

for col in df.columns:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = le

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Encoders saved successfully")