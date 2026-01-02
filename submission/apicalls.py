import requests
import os
import json 
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def call(URL):
        config = json.load(open('config.json', 'r'))

        #Call each API endpoint and store the responses
        response1 = requests.get(URL + 'prediction?filename=testdata/testdata.csv').text
        response2 = requests.get(URL + 'scoring').text
        response3 = requests.get(URL + 'summarystats').text
        response4 = requests.get(URL + 'diagnostics').text

        #combine all API responses
        responses = "prediction=" + str(response1) + '\n'
        responses += "scoring=" + str(response2) + '\n'
        responses += "summarystats=" + str(response3) + '\n'
        responses += "diagnostics=" + str(response4)

        #write the responses to your workspace
        apireturns_filepath = os.path.join(config['output_model_path'], "apireturns.txt")
        with open(apireturns_filepath, 'w') as file:
                file.write(responses)
        logger.info(f"results of API calls saved to {apireturns_filepath}")
        
if __name__ == '__main__':
        #Specify a URL that resolves to your workspace
        URL = 'http://127.0.0.1:8000/'
        call(URL)