from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Accident_Prediction.src.visualize import plot_2d
from Accident_Prediction.src.data_loader import data_loader
from Accident_Prediction.src.preprocess import preprocess_data
from sklearn.decomposition import PCA

def feature_engineering():
    data = preprocess_data()
    print(data.head())
    print("Converting data into format which is processable by model")
    label = LabelEncoder()
    data['Accident_severity'] = label.fit_transform(data['Accident_severity'])
    # df = label.fit_transform(data)
    col_list = data.columns.tolist()
    for item in col_list:
        data[item] = label.fit_transform(data[item])
    pca = PCA(n_components=2)
    df = pca.fit_transform(data)
    print("Feature engineering done")
    y = data['Accident_severity']
    plot_2d(df,y,'Accident_severity')
    dff = data.drop(columns=['Accident_severity'])
    x_train, x_test, y_train, y_test = train_test_split(dff, y, test_size=0.4, random_state=42)
    sm = SMOTETomek(random_state=42)
    x_resample, y_resample = sm.fit_resample(x_train, y_train)
    print("Done with feature engineering")
    return x_resample, y_resample, x_test, y_test

