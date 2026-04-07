import pandas as pd

def data_loader():
    data = pd.read_csv('./data/fsml_data.csv')
    print("Returning data from data_loader")
    return data
