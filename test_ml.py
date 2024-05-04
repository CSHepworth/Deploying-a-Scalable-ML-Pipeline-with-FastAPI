import pytest
import pandas as pd
import os

data = pd.read_csv("./data/census.csv")


# TODO: implement the first test. Change the function name and input as needed
def test_for_nulls():
    # Test for null rows by comparing the shape of the data after dropping nulls
    assert data.shape == data.dropna().shape, "Data contains null values"



# TODO: implement the second test. Change the function name and input as needed
def test_for_duplicates():
    # Test for duplicate rows by checking if the total duplicates is less than 1% of the size of the data
    assert data.duplicated().sum() < data.shape[0], "Data contains more than 1% duplicates"


# TODO: implement the third test. Change the function name and input as needed
def test_model_size():

    # Tests if model.py is smaller than 100mb (100000000 bytes)
    assert os.path.getsize("./model/model.pkl") < 100000000, "model.pkl is too large for github's standards"
