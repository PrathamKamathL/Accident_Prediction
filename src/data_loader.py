import pandas as pd
import kagglehub
import os
import shutil
import logging
def data_loader():
    logging.basicConfig(
        filename='../logs/app.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    path_kaggle = "saurabhshahane/road-traffic-accidents"
    target_dir = "../data"
    path = kagglehub.dataset_download(path_kaggle)
    os.makedirs("data/", exist_ok=True)
    flag = False
    for file_name in os.listdir(path):
        src = os.path.join(path, file_name)
        dst = os.path.join(target_dir, file_name)

        if os.path.isfile(src):
            shutil.copy(src, dst)
            flag = True
    if flag:
        logging.info("Data has been downloaded successfully")
    else:
        logging.info("Data has not been downloaded successfully")



    data = pd.read_csv('../data/RTA Dataset.csv')
    print("Returning data from data_loader")
    return data

if __name__ == "__main__":
    data = data_loader()
    print(data.head())