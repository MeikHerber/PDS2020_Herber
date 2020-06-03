import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .train_test_split import train_test_splitting
from ...input_output import read_file


def prepare_dataset(csv_name='dortmund_preprocessed_regression', dependent_var='duration'):
    # Load dataset
    path_a = Path(os.getcwd()).parent
    path = os.path.join(path_a, f'data\input\{csv_name}.csv')
    if not os.path.exists(path):
        path_a = Path(os.getcwd())
        path = os.path.join(path_a, f'data\input\{csv_name}.csv')
    df = read_file(path)
    # Declare input and output
    x = df.drop(dependent_var, axis=1)
    y = df[dependent_var]
    # Train-test-split
    x_train, x_test, y_train, y_test = train_test_splitting(x, y)
    # Z-standardization
    st_scaler = StandardScaler()
    st_scaler.fit(x_train)
    x_train_scaler = st_scaler.transform(x_train)
    x_test_scaler = st_scaler.transform(x_test)
    x_train_scale = pd.DataFrame(x_train_scaler)
    x_test_scale = pd.DataFrame(x_test_scaler)
    return x_train, x_test, x_train_scale, x_test_scale, y_train, y_test
