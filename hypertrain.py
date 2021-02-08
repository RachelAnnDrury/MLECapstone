import argparse
import joblib
import numpy as np
import pandas as pd
import os

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

run = Run.get_context()

# Set file location
data_location = 'https://raw.githubusercontent.com/RachelAnnDrury/MLECapstone/main/heart_failure_clinical_records_dataset.csv'

# Call and assign file to variable "df"
df = TabularDatasetFactory.from_delimited_files(path=data_location)
# df = pd.read_csv('https://raw.githubusercontent.com/RachelAnnDrury/MLECapstone/main/heart_failure_clinical_records_dataset.csv')

# Clean data from file
def clean_data(data):
    # Scale data
    X_df = data.to_pandas_dataframe().dropna()
    y_df = X_df[X_df.columns['DEATH_EVENT']]
    X_df = X_df.drop(['DEATH_EVENT'], axis = 1)
    X_df = sc.fit_transform(X_df)
#    X = df.drop_columns('DEATH_EVENT')
#    sc = StandardScaler()
#    X_df = sc.fit_transform(X)

#    y_df = X_df.pop('DEATH_EVENT')
    # y_df = df.keep_columns('DEATH_EVENT')
    
    return X_df, y_df

X, y = clean_data(df)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
 
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type = float, default = 1.0, help = "Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type = int, default = 100, help = "Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter = args.max_iter).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok = True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
