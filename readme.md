# Background

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.
![Project: a dynamic risk assessment system
](images/project.png)


# Project Steps Overview

You'll complete the project by proceeding through 5 steps:

1. Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.

2. Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.

3. Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.

4. Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.

5. Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.

# Project Submission

You'll submit a zip file that contains all of the scripts required for the project. Your zip file will also include reports from Step 4 that show the important model metrics for your model.

# The Workspace

You'll complete the project in the Udacity Workspace, which you will find on the Workspace page in the current project lesson. The workspace contains all the computing resources needed to complete the project.

Your workspace has eight locations you should be aware of:

* `/home/workspace`, the root directory. When you load your workspace, this is the location that will automatically load. This is also the location of many of your starter files.
* `/practicedata/`. This is a directory that contains some data you can use for practice.
* `/sourcedata/`. This is a directory that contains data that you'll load to train your models.
* `/ingesteddata/`. This is a directory that will contain the compiled datasets after your ingestion script.
* `/testdata/`. This directory contains data you can use for testing your models.
* `/models/`. This is a directory that will contain ML models that you create for production.
* `/practicemodels/`. This is a directory that will contain ML models that you create as practice.
* `/production_deployment/`. This is a directory that will contain your final, deployed models.

# Starter Files

There are many files in the starter: 10 Python scripts, one configuration file, one requirements file, and five datasets.

The following are the Python files that are in the starter files:

* `training.py`, a Python script meant to train an ML model
* `scoring.py`, a Python script meant to score an ML model
* `deployment.py`, a Python script meant to deploy a trained ML model
* `ingestion.py`, a Python script meant to ingest new data
* `diagnostics.py`, a Python script meant to measure model and data diagnostics
* `reporting.py`, a Python script meant to generate reports about model metrics
* `app.py`, a Python script meant to contain API endpoints
* `wsgi.py`, a Python script to help with API deployment
* `apicalls.py`, a Python script meant to call your API endpoints
* `fullprocess.py`, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed

The following are the datasets that are included in your starter files. Each of them is fabricated datasets that have information about hypothetical corporations.

**Note**: these data have been uploaded to your workspace as well

* `dataset1.csv` and `dataset2.csv`, found in `/practicedata/`
* `dataset3.csv` and `dataset4.csv`, found in `/sourcedata/`
* `testdata.csv`, found in `/testdata/`

The following are other files that are included in your starter files:

* `requirements.txt`, a text file and records the current versions of all the modules that your scripts use
* `config.json`, a data file that contains names of files that will be used for configuration of your ML Python scripts

# Step 1: Data Ingestion

Data ingestion is important because all ML models require datasets for training. Instead of using a single, static dataset, you're going to create a script that's flexible enough to work with constantly changing sets of input files. This step will make your data ingestion go smoothly and easily, even if the data itself is complex.

In this step, you'll read data files into Python, and write them to an output file that will be your master dataset. You'll also save a record of the files you've read.

## Starter Files

For this step, you'll be working with the starter file called ingestion.py as a template for your Python code.

You'll also be working with the config.json configuration file, and the two datasets (dataset1.csv and dataset2.csv) found in /practicedata/.

## Environment Setup (conda example)
* `conda create -n mlproject "python=3.11.14"` (to remove: `rm -rf mlproject`)
* `conda activate mlproject`
* `pip install -r requirements.txt`

## Using config.json Correctly

It's important to understand your config.json starter file since you'll be using it throughout the project. This file contains configuration settings for your project.

This file contains five entries:

* input_folder_path, which specifies the location where your project will look for input data, to ingest, and to use in model training. If you change the value of input_folder_path, your project will use a different directory as its data source, and that could change the outcome of every step.
* output_folder_path, which specifies the location to store output files related to data ingestion. In the starter version of config.json, this is equal to /ingesteddata/, which is the directory where you'll save your ingested data later in this step.
* test_data_path, which specifies the location of the test dataset
* output_model_path, which specifies the location to store the trained models and scores.
* prod_deployment_path, which specifies the location to store the models in production.

When we're initially setting up the project, our config.json file will be set to read practicedata and write ****practicemodels**. **When we're ready to finish the project, you will need to change the locations specified in config.json so that we're reading our actual, sourcedata and we're writing to our models directory.

## Reading Data and Compiling a Dataset

In the first part of your data ingestion.py script, you'll read a collection of csv files into Python.

The location of the csv files you'll be working with is specified in the config.json starter file, in an entry called input_folder_path. In your starter version of config.py, this entry's value is set to the /practicedata/ directory.

You need to add code to your ingestion.py starter file so that it can automatically detect all of the csv files in the directory specified in the input_folder_path. Each of the files in the input_folder_path represents a different dataset. You'll need to combine the data in all of these individual datasets into a single pandas DataFrame.

You shouldn't manually write file names in your script: your script needs to automatically detect every file name in the directory. Your script should work even if we change the number of files or the file names in the input_folder_path.

It's possible that some of the datasets that you read and combine will contain duplicate rows. So, you should de-dupe the single pandas DataFrame you create, and ensure that it only contains unique rows.

## Writing the Dataset

Now that you have a single pandas DataFrame containing all of your data, you need to write that dataset to storage in your workspace. You can save it to a file called finaldata.csv. Save this file to the directory that's specified in the output_folder_path entry of your config.json configuration file. In your starter version of config.json, the output_folder_path entry is set to /ingesteddata/, so your dataset will be saved to /ingesteddata/.

## Saving a Record of the Ingestion

For later steps in the project, you'll need to have a record of which files you read to create your finaldata.csv dataset. You need to create a record of all of the files you read in this step, and save the record on your workspace as a Python list.

You can store this record in a file called ingestedfiles.txt. This file should contain a list of the filenames of every .csv you've read in your ingestion.py script. You can also save this file to the directory that's specified in the output_folder_path entry of your config.json configuration file.
