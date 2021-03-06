# Heart Failure Prediction with Microsoft Azure Machine Learning

Machine Learning Engineer with Microsoft Azure Capstone Project: Heart Failure Prediction (from Kaggle dataset)


## Project Set Up and Installation
The Jupyter Notebooks in this project are designed for use in the Microsoft Azure SDK. Files necessary to upload to the directory include hyperparameter_tuning.ipynb, automl.ipynb, train.py, score.py, and the dataset (heart_failure_clinical_records_dataset.csv, which will be uploaded from the internet within the Notebooks, if needed). Each Jupyter Notebook can be run independently, but it is best to first run the hyperparameter_tuning.ipynb and then the automl.ipynb, since they are assigned to use the same compute, and the automl.ipynb will clean-up resources. To use a different compute than the one assigned, update the amlcompute_cluster_name from 'mlecscompute' to the name of the desired existing compute. Please note that the Notebooks also require interactive authentication to set-up the workspace, which involves entering a key code through Microsoft's device login. 

![Diagram](https://user-images.githubusercontent.com/73516567/107342291-e1ae9e00-6a74-11eb-84a0-021a056d241d.png)

## Dataset

### Overview
*Kaggle Dataset Description*

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

*Kaggle URL:* https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/code

### Acknowledgements
This Jupyter Notebook utilizes the Heart Failure Prediction Dataset, downloaded from Kaggle under License "Attribution 4.0 International" or "CC BY 4.0." Credit: Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020).

### Dataset Column Details
- Age (patient age, in years)
- Anaemia (binary classification)
- Creatinine Phosphokinase (mcg/L of CPK enzyme in patient's blood)
- Diabetes (binary classification)
- Ejection fraction (% of blood exiting heart with each contraction)
- High blood pressure (binary classification)
- Platelets (kiloplatelets/mL in patient's blood)
- Serum creatinine (md/dL of creatinine in patient's blood)
- Serum sodium (mEq/L of sodium in patient's blood)
- Sex (binary classification)
- Smoking (binary classification)
- Time (days included in follow-up period)
- Death event (binary classification; if patient died during follow-up period) *TARGET COLUMN*

### Task
Create a binary classification model to assess the likelihood of a patient's death by heart failure event using the Kaggle dataset. This project includes two separate Jupyter Notebooks: hyperparameter_tuning.ipynb and automl.ipynb, which include a machine learning experiment powered by HyperDrive for hyperparameter tuning, and an AutoML experiment, respectively. The most accurate model from these experiments was from the AutoML experiment, so that model is also deployed to a web service, and the web service is pinged with a test prediction. The automl.ipynb Notebook also includes the necessary resources clean-up to end the experiment. 

### Access
The dataset is from outside the AzureML framework. Both Notebooks search for the dataset to check if it has already been uploaded to Azure Machine Learning Studio under the assigned title and description. If it has been previously uploaded, it will pull the information and print details. If the dataset has not yet been uploaded, the Notebook will find it through the URL and load the file using the designated filename and description. 

## Automated ML
The configuration settings for this AutoML experiment rely on the advantages of using AutoML to detect the relationship between a series of variables and a target binary classification. Settings allow for twenty minutes of experimentation with a maximum of five concurrent iterations focused on increasing the overall accuracy of the model.

**AutoML Parameters:**

*AutoMLSettings*
- Experiment Timeout: 20 minutes
- Maximum Concurrent Iterations: 5
- Primary Metric: Accuracy

*AutoML Config:*
- Task: Classification
- Training Data: "dataset" variable (dataset referenced above)
- Label Column Name (Target): "DEATH_EVENT"
- Path: Project Folder
- Enable Early Stopping Policy: True
- Featurization: Auto
- Debug Log: "automl_errors.log"

![AutoMLSettings](https://user-images.githubusercontent.com/73516567/107456650-60531c00-6b05-11eb-8dae-d8f08bdbcf57.png)

### Results
The most accurate model produced by the AutoML experiment was almost always Voting Ensemble, followed closely by Stacking Ensemble, with accuracy ranging from 87% - 89%. Separate experimental attempts at altering the settings yielded little to no difference, so the best way to improve the accuracy of this model would be to obtain more datapoints (rows in the dataset) and potentially additional relevant features (columns in the dataset). Generally time, serum_creatinine, and ejection fraction were the most important variables, but there may be additional information (features) that could be included about the patients to improve accuracy without overfitting. 

*Screenshot: Selecting RunDetails Widget from AutoML Run by clicking "View run details"*
![AutoMLRunDetailsWidget4](https://user-images.githubusercontent.com/73516567/107461215-67caf300-6b0e-11eb-8cda-7ead3998337c.png)


*Screenshot: RunDetails Widget from AutoML Run*
![AutoMLRunDetailsWidget1](https://user-images.githubusercontent.com/73516567/107421126-7691b580-6ace-11eb-9674-18161b0f7c0a.png)


*Screenshot: RunDetails Widget from AutoML Run with Child Run Details*
![AutoMLRunDetailsWidget2](https://user-images.githubusercontent.com/73516567/107421081-6c6fb700-6ace-11eb-9c5e-1ca7e8bf7a16.png)


*Screenshot: Best Model from AutoML Run*
![AutoMLBestModel](https://user-images.githubusercontent.com/73516567/107420857-37fbfb00-6ace-11eb-959c-e565366d40c1.png)
![AutoMLRunMetrics](https://user-images.githubusercontent.com/73516567/107422577-0e43d380-6ad0-11eb-8b19-ff45c546e71a.png)


## Hyperparameter Tuning
Logistic regression uses a logit function to compute the probability of outcomes with multiple explanatory variables. Logistic Regression can handle sparse input, making it useful for a small dataset, as seen in this project.

The "C" parameter controls the penalty strength ("inverse regularity strength") and the "max_iter" parameter caps the number of iterations taken for solvers to converge. These two parameters provide a big "bang for the buck" when considering which hyperparameters to tune on a small dataset with limited compute time. Random Sampling is easy to use and provides good accuracy of representation, while keeping time and compute resources minimized, and the using BanditPolicy for early termination minimizes opportunity costs while minimizing regret. Altogether, this configuration is designed for computing efficiency while maintaining efficacy.

### Results
The Logistic Regression Model powered by HyperDrive hyperparameter tuning consistently gave results around 81% accuracy (the primary metric), which is above the minimum that I set for the experiment (80%), but below the target of 85%+. Refining and fine tuning additional hyperparameters could have provided a superior model, but reviewing the results, including those from the AutoML run, a random forest classifier would have likely provided a more accurate model. If possible, gathering additional datapoints and expanding the set could also increase the accuracy of the model. 

*Screenshot: RunDetails Widget from Hyperdrive Hyperparameter Tuning Run*
![HyperRunDetailsWidget2](https://user-images.githubusercontent.com/73516567/107421193-85786800-6ace-11eb-8c8c-a5b596bb2ad0.png)


*Screenshot: RunDetails Widget from Hyperdrive Hyperparameter Tuning Run with Child Run Details*
![HyperRunDetailsWidget3](https://user-images.githubusercontent.com/73516567/107421017-595ce700-6ace-11eb-8ef8-1ee69f01ee71.png)


*Screenshot: Best Model from Hyperdrive Hyperparameter Tuning Run*
![HyperBestModel4](https://user-images.githubusercontent.com/73516567/107420913-4518ea00-6ace-11eb-86da-299e68cd063b.png)


## Model Deployment
The deployed Voting Ensemble model takes the input variables in the "Test Web Service" section of the automl.ipynb file. To submit a query, simply edit the "Test Web Service" data to the desired inputs and run that cell and the following, which contains the "result" variable and prints the prediction based upon the data inputs. 

***SCREENSHOT*** Deployed Model - test (from notebook)
*Screenshot: Deployed Model*
![AutoMLEndpoint1](https://user-images.githubusercontent.com/73516567/107421230-8e693980-6ace-11eb-99ff-e97f60387263.png)


*Screenshot: Web Service Test on Deployed Model (binary prediction result = 1; positive DEATH_EVENT)*
![TestWebService](https://user-images.githubusercontent.com/73516567/107422607-1734a500-6ad0-11eb-93a9-2cd6e4d54326.png)


## Future Work
The AutoML model produced in this project had less than 90% accuracy and leaves room for improvement. The following are some suggestions that could increase the accuracy of the model:

- Increase training data (the dataset is rather small, with only 300 rows); additional datapoints could improve accuracy or better reflect the accuracy of a model in a larger sample size. 
- Provide additional input variables (there may be other factors contributing to patient deaths outside the provided variables; using machine learning to determine the relevance of additional data could provide a more accurate and relevant model).
- Increase resource allocation and add layers if more data is available to analyze (at the current size, additional resources would not be well-used, but if new variables were added, additional computing time and resources may be necessary to analyze their relevance in relation to the existing variables and each other. 

## Screen Recording
Visit this URL: https://youtu.be/XPYkFoh2UFw to view the project screencast, demonstrating: 
- A working model
- Demo of the deployed  model
- Sample request sent to the endpoint and its response
