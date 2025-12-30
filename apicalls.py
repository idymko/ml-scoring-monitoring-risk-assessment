import requests
import os
import json 

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

config = json.load(open('config.json', 'r'))

#Call each API endpoint and store the responses
response1 = requests.get('http://127.0.0.1:8000/prediction?filename=testdata/testdata.csv').content
response2 = requests.get('http://127.0.0.1:8000/scoring').content
response3 = requests.get('http://127.0.0.1:8000/summarystats').content
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content

#combine all API responses
responses = str(response1) + "\n"
responses += str(response2) + "\n"
responses += str(response3) + "\n"
responses += str(response4)

#write the responses to your workspace
with open(os.path.join(config['output_model_path'], "apireturns.txt"), 'w') as file:
        file.write(responses)
