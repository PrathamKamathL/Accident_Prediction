from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import logging

from src.features import feature_engineering


def training_decisionTree(featureEngineeredData):
    logging.basicConfig(
        filename='../logs/app.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # x_train, y_train, x_test, y_test = feature_engineering()
    data = featureEngineeredData
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    logging.info("Received final data for DecisionTree")
    print("Received final data for DecisionTree")
    decision = DecisionTreeClassifier(random_state=42, max_depth=10)
    decision.fit(x_train, y_train)
    pickle.dump(decision, open('../models/model_v1.pkl', 'wb'))
    return decision

def training_randomForest(featureEngineeredData):
    # x_train, y_train, x_test, y_test = feature_engineering()
    data = featureEngineeredData
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    logging.info("Received final data for RandomForest")
    print("Received final data for RandomForest")
    ensemble = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    ensemble.fit(x_train, y_train)
    pickle.dump(ensemble, open('../models/model_v2.pkl', 'wb'))
    return ensemble

def training_xgboost(featureEngineeredData):
    # x_train, y_train, x_test, y_test = feature_engineering()
    logging.basicConfig(
        filename='../logs/app.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    data = featureEngineeredData
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    print("Received final data for XGBoost")
    xgb = XGBClassifier(random_state=42)
    xgb.fit(x_train, y_train)
    pickle.dump(xgb, open('../models/model_v3.pkl', 'wb'))
    return xgb
