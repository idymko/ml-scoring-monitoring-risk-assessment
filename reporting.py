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

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # return list of prediction by model in "prod_deployment_path" on data in "test_data_path"
    config = json.load(open('config.json', 'r'))
    X = pd.read_csv(os.path.join(config['test_data_path'], 'testdata.csv'), index_col=0)
    y = X.pop("exited")
    preds = model_predictions(os.path.join(config['test_data_path'], 'testdata.csv'))

    # create confusion matrix
    confusion_matrix_path = config['output_model_path']
    
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    cm = metrics.confusion_matrix(
                y_true=y,
                y_pred=preds,
                normalize="true"
            )

    disp  = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(
        ax=sub_cm,
        values_format=".1f",
        xticks_rotation=90,
    )

    fig_cm.tight_layout()
    plt.savefig(os.path.join(confusion_matrix_path, 'confusionmatrix.png'))
    plt.close

if __name__ == '__main__':
    score_model()
