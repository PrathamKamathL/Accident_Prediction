from src.data_loader import data_loader
from src.preprocess import preprocess_data
from src.features import feature_engineering
from src.train import *
from src.evaluate import *
import logging
def ml_pipeline():
    rawdata = data_loader()
    logging.basicConfig(
        filename='../logs/app.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    preprocessed_data = preprocess_data(rawdata)
    logging.info("Preprocessing Successful")
    featureEngineeredData = feature_engineering(preprocessed_data)
    logging.info("Feature Engineering Successful")
    model_v1 = training_decisionTree(featureEngineeredData)
    logging.info("Training Successful for decision tree")
    model_v2 = training_randomForest(featureEngineeredData)
    logging.info("Training Successful for random forest")
    classification_report_decision_tree(model_v1, featureEngineeredData)
    logging.info("Classification Report Successful for decision tree")
    classification_report_random_forest(model_v2, featureEngineeredData)
    logging.info("Classification Report Successful for random forest")
    print("Pipeline completed execution successfully")

if __name__ == "__main__":
    ml_pipeline()