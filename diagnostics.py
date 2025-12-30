
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

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

# ##################Function to get timings
# def execution_time():
#     #calculate timing of training.py and ingestion.py
#     return #return a list of 2 timing values in seconds

# ##################Function to check dependencies
# def outdated_packages_list():
#     #get a list of 


if __name__ == '__main__':
    print(model_predictions())
    print(dataframe_summary())
    # execution_time()
    # outdated_packages_list()





    
