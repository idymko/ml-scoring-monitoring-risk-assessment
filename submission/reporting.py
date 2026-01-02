import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

##############Function for reporting
def plot_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model.
    Write the confusion matrix to the workspace.
    """
    
    # return list of prediction by model in "prod_deployment_path" on data in "test_data_path"
    config = json.load(open('config.json', 'r'))
    X = pd.read_csv(os.path.join(config['test_data_path'], 'testdata.csv'), index_col=0)
    y = X.pop("exited").values
    preds = model_predictions(X)

    if len(y)<10:
        logger.info(f"y_true={y}")
        logger.info(f"y_pred={preds}")
    else:
        logger.info(f"labels not printed as their legth > 10")
        
    # create confusion matrix (cm)
    fig_cm, sub_cm = plt.subplots(figsize=(5, 5))
    sub_cm.set_title('Confusion matrix')
    cm = metrics.confusion_matrix(y_true=y,y_pred=preds)
    disp  = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=sub_cm,colorbar=False)
    fig_cm.tight_layout()
    
    # save cm to output_model_path
    plt.savefig(os.path.join(config['output_model_path'], 'confusionmatrix.png'))
    plt.close

if __name__ == '__main__':
    plot_confusion_matrix()
