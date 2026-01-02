import json
import os
import ast
import pickle
import ingestion
import training
import scoring
import deployment
import apicalls
import reporting
import logging 

log_path = os.path.dirname(os.path.realpath(__file__))
log_file = os.path.join(log_path, 'fullprocess.log')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') )
logger.addHandler(file_handler)

URL = 'http://127.0.0.1:8000/'

def fullprocess():
    
    #config = json.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config.json'), 'r'))
    config = json.load(open('config.json', 'r'))
    
    # Check if any deployment has been done
    deplayedfiles = os.listdir(config['prod_deployment_path'])
    expectedfiles = ['trainedmodel.pkl', 'ingestedfiles.txt', 'latestscore.txt']
    first_deployment = False
    for deployedfile in expectedfiles:
        if deployedfile not in deplayedfiles:
            first_deployment = True
    
    logger.info(f"first_deployment={first_deployment}")
    
    # Check and read new data
    ## First, read ingestedfiles.txt from the deployment directory prod_deployment_path, 
    if not first_deployment:
        with open(os.path.join(config["prod_deployment_path"],"ingestedfiles.txt"),'r') as file:
            ingestedfiles = file.read()

        ### split by EoL
        ingestedfiles = ingestedfiles.split("\n")
        ### split path into directories and filenames
        ingestedfiles = [os.path.split(path) for path in ingestedfiles]
        ingestedfiles = [item for tup in ingestedfiles for item in tup]
        ### consider only filenames ending with .csv
        ingestedfiles = [ingestedfile for ingestedfile in ingestedfiles if ingestedfile.endswith(".csv")]
    else:
        ingestedfiles = []
    
    ## Second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    filenames = os.listdir(config['input_folder_path'])
    filenames = [filename for filename in filenames if filename.endswith(".csv")]

    ### find elements in filenames that do not exist in ingestedfiles
    newfiles = list(set(filenames) - set(ingestedfiles))
    
    ## Decide whether to proceed (part 1)
    ## if you found new data, you should proceed. otherwise, do end the process here
    if len(newfiles)>0:
        logger.info(f"PROCEED: new data files found={newfiles}")
        ingestion.merge_multiple_dataframe(newfiles)
    else:
        logger.info("STOP: no new data files found")
        return None
    
    # Checking for model drift
    ## Check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if not first_deployment:
        with open(os.path.join(config['prod_deployment_path'],'latestscore.txt')) as file:
            latestscore = ast.literal_eval(file.read())
        
        logger.info(f"latestscore={latestscore}")
        
        newscore = scoring.score_model('prod')
        logger.info(f"newscore={newscore}")
    
        ## Deciding whether to proceed (part 2)
        ### if you found model drift, you should proceed. otherwise, do end the process here

        if newscore < latestscore:
            logger.info("PROCEED: new model score has decreased according to the raw test")
        else:
            logger.info("STOP: new model score is higher in raw test")
            return None

    logger.info(f"training new model")
    training.train_model()
    logger.info(f"scoring new model")
    scoring.score_model()

    # Re-deployment
    logger.info(f"deploying new model")
    deployment.deploy_model()
    
    
    # Diagnostics and reporting
    logger.info(f"running reporting and APIs")
    reporting.plot_confusion_matrix()
    apicalls.call(URL)

if __name__ == "__main__":
    fullprocess()