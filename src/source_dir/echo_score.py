# Load libraries
import base64 
import json
import os 
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# 1. Requried init function
def init():
    # Create a global variable for loading the model
    global model
    modelFile = "linearRegressorModel.joblib"
    model = joblib.load(os.path.join(os.getenv("AZUREML_MODEL_DIR"), modelFile))

# 2. Requried run function
def run(request):
    # Receive the data and run model to get predictions 
    data = json.loads(request)

    if 'base64' in data.keys():
        if type(data["data"]) == str:
            data = str(base64.b64decode(data["data"]), 'utf-8')
        elif type(data["data"]) == bytes:
            data = data["data"].decode('utf-8')
        else:
            pass
    else:
        data = data["data"]
    final_test = np.array([data], dtype='object')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',max_features=10, strip_accents='unicode')
    vectorizer.fit_transform(final_test).toarray()
    final_test_vector = vectorizer.transform(final_test).toarray()
    res = model.predict(final_test_vector)
    return np.array2string(res, precision=3)[2:-2]