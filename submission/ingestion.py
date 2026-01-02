import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#############Function for data ingestion
def merge_multiple_dataframe(filenames = []):
    
    #############Load config.json and get input and output paths    
    config = json.load(open('config.json', 'r'))
    
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    
    #check for datasets, compile them together, and write to an output file
    if filenames == []:
        filenames = os.listdir(input_folder_path)
    
    ingestedfiles_content = ""
    for idx, filename in enumerate(filenames):
        
        filepath = os.path.join(input_folder_path, filename)
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
    data = data.drop_duplicates().reset_index(drop=True)
    
    # save to dataframe to csv file
    ingesteddata_filepath = os.path.join(output_folder_path, 'finaldata.csv')
    data.to_csv(ingesteddata_filepath, index=False)
    logger.info(f"ingested data saved '{ingesteddata_filepath}'")

    ingestedfiles_filepath = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingestedfiles_filepath, 'w') as file:
        file.write(ingestedfiles_content)
    logger.info(f"list of ingested files saved to '{ingestedfiles_filepath}'")

if __name__ == '__main__':
    merge_multiple_dataframe()
