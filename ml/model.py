import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelBinarizer
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array) -> RandomForestClassifier:

    model = RandomForestClassifier(n_estimators = 200)
    model.fit(X_train, y_train)
    return model



def compute_model_metrics(y: np.array, preds: np.array):

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.array) -> np.array:

    preds = model.predict(X)
    return preds

def save_model(model: RandomForestClassifier, path: str):

    pickle.dump(model, open(path, "wb"))

def load_model(path: str):
    
    file = pickle.load(open(path, "rb"))
    return file


def performance_on_categorical_slice(
    data: pd.DataFrame, 
    column_name: str, 
    slice_value: str, 
    categorical_features: list, 
    label: str, 
    encoder: OneHotEncoder, 
    lb: LabelBinarizer, 
    model: RandomForestClassifier
):

    X_slice, y_slice, _, _ = process_data(
        # your code here
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
        X = data[data[column_name] == slice_value], 
        categorical_features = categorical_features,
        label = label,
        training = False,
        encoder = encoder,
        lb = lb
    )
    preds = inference(model, X_slice)# your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
