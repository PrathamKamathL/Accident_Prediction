from src.data_loader import data_loader
from src.preprocess import preprocess_data
from src.features import feature_engineering
from src.train import *
from src.evaluate import *

def ml_pipeline():
    rawdata = data_loader()
    preprocessed_data = preprocess_data(rawdata)
    featureEngineeredData = feature_engineering(preprocessed_data)
    model_v1 = training_decisionTree(featureEngineeredData)
    model_v2 = training_randomForest(featureEngineeredData)
    classification_report_decision_tree(model_v1, featureEngineeredData)
    classification_report_random_forest(model_v2, featureEngineeredData)
    print("Pipeline completed execution successfully")

if __name__ == "__main__":
    ml_pipeline()