import pandas as pd

data = pd.read_csv('../data/fsml_data.csv')
print("First 5 elements from dataset:")
print(data.head())

def data_loader():
    data = pd.read_csv('../data/fsml_data.csv')
    return data
