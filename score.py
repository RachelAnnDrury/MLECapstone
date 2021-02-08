import joblib
import json
import numpy as np
import pandas as pd
import pickle
import os

from azureml.core import Model

def init():
    global model
    # Change 'registered.sav' to match AutoML
    model_path = Model.get_model_path('registered.sav')
    model = joblib.load(model_path)

def run(data):
    try:
        dload = json.loads(data)
        data = pd.DataFrame(dload['data'])
        result = model.predict(data)
        
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
