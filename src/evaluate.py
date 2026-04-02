from sklearn.metrics import classification_report, confusion_matrix
from Accident_Prediction.src.train import *
from Accident_Prediction.src.features import *
def classification_report_decision_tree():
    model = training_decisionTree()
    _,_,x_test, y_test = feature_engineering()
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

def classification_report_random_forest():
    model = training_RandomForest()
    _,_,x_test, y_test = feature_engineering()
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

def classification_report_xgboost():
    model = training_xgboost()
    _,_,x_test, y_test = feature_engineering()
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
