import sklearn
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from ...input_output import save_model
from ...input_output import load_performance_classification
from ...input_output import save_performance_classification
from ..general import sample_fraction
import pandas as pd
import os
from pathlib import Path
import itertools
from sklearn.linear_model import LogisticRegression


def modelling(model_name, x_train, x_test, y_train, y_test, fraction=1, export=False, leaf_size=30, n_neighbors=5,
              n_estimators=100, min_samples_split=2, min_samples_leaf=1, layers=32, batch_size=32, verbose=0):
    x_sample, y_sample = sample_fraction(fraction, x_train, y_train)  # Create a fraction of the train set
    start_time = time.time()
    model = model_selection(model_name, x_sample, y_sample, y_test, leaf_size, n_neighbors, n_estimators,
                            min_samples_split, min_samples_leaf, layers, batch_size, verbose)  # Train model
    end_time = time.time()
    lapsed_time = end_time - start_time  # Shows how long modelling takes
    if export: save_model(model, model_name)  # Save model
    if model_name != 'Mean':
        y_predict = model.predict(x_test)
        if model_name == 'Neural_Network_1hl' or model_name == 'Neural_Network_2hl':
            y_predict = list(itertools.chain.from_iterable(y_predict))  # Transform double list to single list
            y_predict = pd.DataFrame(y_predict)
            y_predict = y_predict.iloc[:, 0].apply(lambda x: round(x))  # Get values of 0 or 1
    else:
        y_predict = model
    performance = load_performance_classification()  # Load all performance statistics
    save_performance_classification(performance, model_name, y_test, y_predict, lapsed_time,
                                    fraction)  # Update performance statistics


def model_selection(name, x_train, y_train, y_test, leaf_size, n_neighbors, n_estimators, min_samples_split,
                    min_samples_leaf, layers, batch_size, verbose):
    if name == "Mean":
        df = pd.DataFrame(y_test)
        df['towards_uni'] = round(y_train.mean())
        return df['towards_uni']
    elif name == 'Logistic_Regression':
        return sklearn.linear_model.LogisticRegression().fit(x_train, y_train)
    elif name == 'Random_Forest_Classification':
        return sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf).fit(x_train, y_train)
    elif name == 'KNN_Classification':
        return KNeighborsClassifier(leaf_size=leaf_size, n_neighbors=n_neighbors).fit(x_train, y_train)
    elif name == 'Neural_Network_1hl':
        return neural_network(1, layers, batch_size, x_train, y_train, verbose)
    elif name == 'Neural_Network_2hl':
        return neural_network(2, layers, batch_size, x_train, y_train, verbose)


def neural_network(hidden_layers, layers, batch_size, x_train, y_train, verbose=0):
    path_a = Path(os.getcwd()).parent
    path = os.path.join(path_a, r'data\output\best_model_classification.h5')
    model = Sequential()
    model.add(Dense(layers, activation='relu', input_dim=x_train.shape[1]))
    i = hidden_layers - 1
    while i > 0:
        model.add(Dense(layers, activation='relu'))
        i = i - 1
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True)]
    model.fit(x_train, y_train, epochs=10000, batch_size=batch_size, validation_split=0.3, callbacks=callbacks,
              verbose=verbose)
    return load_model(path)
