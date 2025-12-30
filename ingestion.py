import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Function for data ingestion
def merge_multiple_dataframe(config_file = None):
    
    
    #############Load config.json and get input and output paths
    if config_file == None:
        config_file = 'config.json'
        
    with open(config_file,'r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(input_folder_path)
    ingestedfiles_content = ""
    for idx, filename in enumerate(filenames):
        
        filepath = input_folder_path + '/' +  filename
        if filepath.endswith(".csv"):
            ingestedfiles_content += filepath + "\n"
            current_data = pd.read_csv(filepath)
            
            if idx == 0:
                # read first file into new dataframe
                data = current_data.copy()
            else:
                # append all next files
                data = pd.concat([data, current_data], ignore_index=True)        
    
    # drop duplicates
    data.drop_duplicates().reset_index(drop=True)
    
    # save to dataframe to csv file
    filepath = output_folder_path + '/finaldata.csv'
    data.to_csv(filepath, index=False)
    print(f"combined data saved to {filepath}")

    with open(output_folder_path + '/ingestedfiles.txt', 'w') as file:
        file.write(ingestedfiles_content)


if __name__ == '__main__':
    merge_multiple_dataframe()
