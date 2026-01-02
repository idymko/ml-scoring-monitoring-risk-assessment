from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, missing_data
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

#######################Prediction Endpoint
@app.route("/prediction")
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    df = pd.read_csv(filename)
    # X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    X = df.drop(['corporation', 'exited'], axis=1)
    preds = model_predictions(X)
    return str(preds) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring")
def scoring():        
    #check the score of the deployed model
    f1score = score_model()
    return str(f1score) #add return value (a single F1 score number)
    
#######################Summary Statistics Endpoint
@app.route("/summarystats")
def summarystats():        
    #check means, medians, and modes for each column
    df_stats = dataframe_summary()
    return str(df_stats) #return a list of all calculated summary statistics
    
#######################Diagnostics Endpoint
@app.route("/diagnostics")
def diagnostics():    
    #check timing and percent NA values
    exe_time = execution_time()
    nan_percentage = missing_data()
    outdated_packages = outdated_packages_list()
    
    return_dict = {
                    "execution_time": exe_time,
                    "missing_data": nan_percentage,
                    "outdated_packages": outdated_packages
                    }
    
    return str(return_dict) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
