
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################Function to get model predictions
def model_predictions(data):
    """
    Read the deployed model and a dataset from filename, calculate predictions
    
    Inputs:
        data (Pandas) : input dataframe to make predictions on
    Returs:
        preds (list) : list of predictions
    """
    
    config = json.load(open('config.json', 'r'))    
    model = pickle.load(open(
        os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl'), 
        'rb'))
    preds = model.predict(data)
    
    return preds #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here: means, medians, and standard deviations
    # This function should calculate these summary statistics for the 
    # dataset stored in the directory specified by output_folder_path 
    # in config.json. It should output a Python list containing all of 
    # the summary statistics for every numeric column of the input dataset.
    config = json.load(open('config.json', 'r'))
    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    data = pd.read_csv(dataset_csv_path, index_col=0)
    
    means_array= data.mean(axis=0, numeric_only=True)
    medians_array= data.median(axis=0, numeric_only=True)
    std_array= data.std(axis=0, numeric_only=True)
    
    summary_dict = {}
    for idx, column_name in enumerate(means_array.index):
        summary_dict[column_name] = {"mean": means_array[idx], 
                                "median": medians_array[idx], 
                                "std": std_array[idx]}
    
    return summary_dict

def missing_data():
    """
    Function to check for missing data. By missing data, we mean NA values. 
    Remember that the Pandas module has a custom method for checking whether a value is NA.

    Your function needs to count the number of NA values in each column of your dataset. 
    Then, it needs to calculate what percent of each column consists of NA values.

    The function should count missing data for the dataset stored 
    in the directory specified by output_folder_path in config.json. 
    It will return a list with the same number of elements as the number of columns in your dataset. 
    Each element of the list will be the percent of NA values in a particular column of your data.
    """
    config = json.load(open('config.json', 'r'))
    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    data = pd.read_csv(dataset_csv_path, index_col=0)
    missing_data_array = data.isna().sum()/len(data)*100
    
    missing_data_dict = {}
    for column_name, value in missing_data_array.items():
        missing_data_dict[column_name] = value
    
    # return list(data.isna().sum()/len(data)*100)
    return missing_data_dict
        

#################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    logger.info(f"Calculating execution time of 'ingestion.py'")
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer()-start_time
    
    logger.info(f"Calculating execution time of 'training.py'")
    start_time = timeit.default_timer()
    os.system('python3 training.py')
    trainig_time = timeit.default_timer()-start_time
    
    exe_time_dict = {"ingestion.py": ingestion_time, "training.py": trainig_time}
    return exe_time_dict #list([ingestion_time, trainig_time])

##################Function to check dependencies
def outdated_packages_list():
    """
    function that checks the current and latest versions of all the modules 
    that your scripts use (the current version is recorded in requirements.txt). 
    It will output a table with three columns: the first column will show 
    the name of a Python module that you're using; the second column will show 
    the currently installed version of that Python module, 
    and the third column will show the most recent available version of that Python module.
    """
    outdated_packages = subprocess.check_output(['python', '-m', 'pip', 'list', '--outdated'], 
                                                stderr=subprocess.STDOUT).decode().splitlines()[2:]
    
    list_outdated_packages = {}
    for package in outdated_packages:
        package_info = package.split()
        #list_outdated_packages += [package_info[0], package_info[1],package_info[2]]
        list_outdated_packages[package_info[0]] = {"version": package_info[1], "latest": package_info[2]}
    return list_outdated_packages

if __name__ == '__main__':
    
    config = json.load(open('config.json', 'r'))
    df = pd.read_csv(os.path.join(config['test_data_path'], 'testdata.csv'))
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    preds = model_predictions(X)
    logger.info(f"predictions={preds}")
    
    df_summary = dataframe_summary()
    logger.info(f"data_summary={df_summary}")
    
    nan_percentage = missing_data()
    logger.info(f"missing_data={nan_percentage}")
    
    exe_time = execution_time()
    logger.info(f"execution_time={exe_time}")
    
    outdated_packages = outdated_packages_list()
    logger.info(f"outdated_packages={outdated_packages}")   
