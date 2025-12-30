from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    #################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['output_model_path']) 
    test_data_path = os.path.join(config['test_data_path']) 
    
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'), index_col=0)
    y = X.pop("exited")
    preds = model.predict(X)
    f1score = metrics.f1_score(y,preds)
    
    with open(os.path.join(model_path, "latestscore.txt"), 'w') as file:
        file.write(str(f1score))

if __name__ == '__main__':
    score_model()