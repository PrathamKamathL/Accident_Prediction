from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

from crud_operation.Accident_Prediction.src.features import feature_engineering


def training_decisionTree():
    x_train, y_train, x_test, y_test = feature_engineering()
    decision = DecisionTreeClassifier(random_state=42, max_depth=10)
    decision.fit(x_train, y_train)
    pickle.dump(decision, open('../models/model_v1.pkl', 'wb'))
    return decision

def training_RandomForest():
    x_train, y_train, x_test, y_test = feature_engineering()
    ensemble = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    ensemble.fit(x_train, y_train)
    pickle.dump(ensemble, open('../models/model_v2.pkl', 'wb'))
    return ensemble

def training_xgboost():
    x_train, y_train, x_test, y_test = feature_engineering()
    xgb = XGBClassifier(random_state=42)
    xgb.fit(x_train, y_train)
    pickle.dump(xgb, open('../models/model_v3.pkl', 'wb'))
    return xgb

