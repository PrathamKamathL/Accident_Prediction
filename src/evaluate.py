from sklearn.metrics import classification_report, confusion_matrix
from src.train import *
from src.features import *
import logging
def classification_report_decision_tree(model_v1, featureEngineeredData):
    x_test = featureEngineeredData["x_test"]
    y_test = featureEngineeredData["y_test"]
    model = model_v1
    y_pred = model.predict(x_test)
    logging.basicConfig(
        filename='../logs/app.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Classification Report")
    logging.info(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def classification_report_random_forest(model_v2, featureEngineeredData):
    x_test = featureEngineeredData["x_test"]
    y_test = featureEngineeredData["y_test"]
    model = model_v2
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

def classification_report_xgboost():
    model = training_xgboost()
    _,_,x_test, y_test = feature_engineering()
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
