import pandas as pd
import os
import pickle
from pathlib import Path

path_a = Path(os.getcwd()).parent
save_path = os.path.join(path_a, f'data\output')
if not os.path.exists(save_path):
    path_a = Path(os.getcwd())


def read_file(path):
    try:
        df = pd.read_csv(path)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model(name):
    path = os.path.join(path_a, f"data/output/{name}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def load_performance_regression():
    path = os.path.join(path_a, f'data\output\performance_regression.csv')
    if os.path.exists(path):
        performance = read_file(path)
    else:
        performance = pd.DataFrame(columns=['Name', 'MSE', 'CI_low', 'CI_high', 'Time'])
    return performance


def load_performance_classification():
    path = os.path.join(path_a, f'data\output\performance_classification.csv')
    if os.path.exists(path):
        performance = read_file(path)
    else:
        performance = pd.DataFrame(columns=['Name', 'Accuracy', 'CI_low', 'CI_high', 'Time'])
    return performance



