from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

def copy_file(source_file, destination_file):
    with open(source_file, 'rb') as src, open(destination_file, 'wb') as dest:
        dest.write(src.read())
    print(f"copied {source_file} to {destination_file}")

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, 
    # and the ingestedfiles.txt file into the deployment directory
        
    ##################Load config.json and correct path variable
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['output_model_path']) 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    prod_deployment_path = os.path.join(config['prod_deployment_path'])         
        
    # copy model
    copy_file(model_path + '/trainedmodel.pkl', prod_deployment_path + '/trainedmodel.pkl')
    # model = pickle.load(open(model_path + '/trainedmodel.pkl', 'rb'))
    # pickle.dump(model, open(prod_deployment_path + '/trainedmodel.pkl', 'wb'))

    # copy latestscore.txt
    copy_file(model_path + '/latestscore.txt', prod_deployment_path + '/latestscore.txt')
    # with open(model_path + '/latestscore.txt', 'rb') as src, open(prod_deployment_path + '/latestscore.txt', 'wb') as dest:
    #     dest.write(src.read())
        
    # copy ingestedfiles.txt
    copy_file(dataset_csv_path + '/ingestedfiles.txt', prod_deployment_path + '/ingestedfiles.txt')
    # with open(dataset_csv_path + '/ingestedfiles.txt', 'rb') as src, open(prod_deployment_path + '/ingestedfiles.txt', 'wb') as dest:
    #     dest.write(src.read())
        
if __name__ == '__main__':
    store_model_into_pickle()