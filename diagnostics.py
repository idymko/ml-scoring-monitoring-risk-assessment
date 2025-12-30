
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    with open('config.json','r') as f:
        config = json.load(f)
    test_data_path = os.path.join(config['test_data_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'), index_col=0)
    y = X.pop("exited")
    preds = model.predict(X)
    
    if len(y) != len(preds):
        raise Exception("Len of y is not equal to len of preds!")
    
    return preds #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here: means, medians, and standard deviations
    # This function should calculate these summary statistics for the 
    # dataset stored in the directory specified by output_folder_path 
    # in config.json. It should output a Python list containing all of 
    # the summary statistics for every numeric column of the input dataset.
    with open('config.json','r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    data = pd.read_csv(dataset_csv_path, index_col=0)
    summary_stat = list(data.mean(axis=0))
    summary_stat += list(data.median(axis=0))
    summary_stat += list(data.std(axis=0))
    
    return summary_stat #return value should be a list containing all summary statistics

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
    with open('config.json','r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    data = pd.read_csv(dataset_csv_path, index_col=0)
    
    return list(data.isna().sum()/len(data)*100)
        

#################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer()-start_time
    
    start_time = timeit.default_timer()
    os.system('python3 training.py')
    trainig_time = timeit.default_timer()-start_time
    
    return list([ingestion_time, trainig_time])

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
    
    outdated_packages_df = pd.DataFrame(columns=['Package', 'Version', 'Latest'])
    for package in outdated_packages:
        
        package_info = package.split()
        package_name = package_info[0]
        current_version = package_info[1]
        latest_version = package_info[2]
        new_row = pd.DataFrame([[package_info[0], package_info[1],package_info[2]]], 
                                columns=outdated_packages_df.columns)
        outdated_packages_df = pd.concat([outdated_packages_df,new_row], ignore_index=True)
            
    
    return outdated_packages_df

if __name__ == '__main__':
    preds = model_predictions()
    print(f"model predictions: {preds}")
    
    df_summary = dataframe_summary()
    print(f"data summary (mean, median, std) for each column: {df_summary}")
    
    nan_percentage = missing_data()
    print(f"percentage of NaN per column: {nan_percentage}")
    
    exe_time = execution_time()
    print(f"execution time (ingest, train): {exe_time}")
    
    outdated_packages = outdated_packages_list()
    print(f"outdated packages\n{outdated_packages}")   
