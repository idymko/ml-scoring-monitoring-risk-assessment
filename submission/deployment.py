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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def copy_file(source_file, destination_file):
    with open(source_file, 'rb') as src, open(destination_file, 'wb') as dest:
        dest.write(src.read())
    logger.info(f"deploy '{source_file}' to '{destination_file}'")

####################function for deployment
def deploy_model():
    # copy the latest pickle file, the latestscore.txt value, 
    # and the ingestedfiles.txt file into the deployment directory
        
    ##################Load config.json and correct path variable
    config = json.load(open('config.json', 'r'))

    model_path = os.path.join(config['output_model_path']) 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
        
    # copy model
    copy_file(os.path.join(model_path, 'trainedmodel.pkl'), 
              os.path.join(prod_deployment_path, 'trainedmodel.pkl'))

    # copy latestscore.txt
    copy_file(os.path.join(model_path, 'latestscore.txt'), 
              os.path.join(prod_deployment_path, 'latestscore.txt'))
        
    # copy ingestedfiles.txt
    copy_file(os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
              os.path.join(prod_deployment_path, 'ingestedfiles.txt'))
        
if __name__ == '__main__':
    deploy_model()