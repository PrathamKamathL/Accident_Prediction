from Accident_Prediction.src.data_loader import data_loader

def preprocess_data():
  rawdata = data_loader()
  rawdata.drop(columns=['Day_of_week','Owner_of_vehicle','Service_year_of_vehicle', 'Defect_of_vehicle','Road_allignment','Number_of_vehicles_involved', 'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality', 'Fitness_of_casuality'
                       'Time'], inplace=True)
    #Dropping examples i.e rows which have null
  rawdata.dropna(inplace=True)
  return rawdata
