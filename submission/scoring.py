from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging 

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#################Function for model scoring
def score_model(process='test'):
    
    """
    This function should take a trained model, load test data, 
    and calculate an F1 score for the model relative to the test data
    It should write the result to the latestscore.txt file
    Inputs:
        process : str
        - 'train' - model: output_model_path, data: output_folder_path, save score: no
        - 'test' (default)  - model: output_model_path, data: test_data_path, save score: yes to output_model_path
        - 'prod' - model: prod_deployment_path, data: output_folder_path, save score: no
    Returns:
        f1score : float
    """
    
    #################Load config.json and get path variables
    config = json.load(open('config.json', 'r'))
    if (process=='train'):
        model_filepath = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        data_filepath = os.path.join(config['output_folder_path'], 'finaldata.csv')
    elif (process=='test'):
        model_filepath = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        data_filepath = os.path.join(config['test_data_path'], 'testdata.csv')
    elif (process=='prod'):
        model_filepath = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
        data_filepath = os.path.join(config['output_folder_path'], 'finaldata.csv')
    else:
        raise ValueError(f"Uknown process parameter: {process}")
        
    logger.info(f"scoring model '{model_filepath}' on '{process}' data '{data_filepath}'")
    model = pickle.load(open(model_filepath, 'rb'))
    X = pd.read_csv(data_filepath, index_col=0)
    y = X.pop("exited").values
    preds = model.predict(X)
    f1score = metrics.f1_score(y,preds)
    logger.info(f"f1score={f1score}")
    
    if (process=='test'):
        score_filepath = os.path.join(config['output_model_path'], "latestscore.txt")
        with open(score_filepath,'w') as file:
            file.write(str(f1score))
        logger.info(f"f1score saved to file '{score_filepath}'")

    return f1score

if __name__ == '__main__':
    score_model()