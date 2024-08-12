# load_data.py
from sklearn.datasets import fetch_california_housing

def get_data():
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target
    return X, y
