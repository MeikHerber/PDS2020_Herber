from .sample_fraction import sample_fraction
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.ensemble
import time


def hyperparameter(model_name, x_train, y_train, leaf_size=30, n_neighbors=5, n_estimators=100, min_samples_split=2,
                   min_samples_leaf=1, fraction=1, verbose=0):
    start_time = time.time()
    x_train_sample, y_train_sample = sample_fraction(fraction, x_train, y_train)
    model, hyperparameters, scoring = randomized_search(model_name, leaf_size, n_neighbors, n_estimators,
                                                        min_samples_split, min_samples_leaf)
    best_model = RandomizedSearchCV(model, hyperparameters, cv=5, scoring=scoring, n_iter=100, verbose=verbose).fit(
        x_train_sample, y_train_sample)
    end_time = time.time()
    lapsed_time = end_time - start_time  # Shows how long modelling takes
    print(best_model.best_estimator_.get_params(), 'Time: ', lapsed_time, 'seconds')


def randomized_search(name, leaf_size, n_neighbors, n_estimators, min_samples_split, min_samples_leaf):
    if name == "KNN_Regression":
        return KNeighborsRegressor(), dict(leaf_size=leaf_size, n_neighbors=n_neighbors), 'neg_mean_squared_error'
    elif name == 'Random_Forest_Regression':
        return sklearn.ensemble.RandomForestRegressor(), dict(n_estimators=n_estimators,
                                                              min_samples_split=min_samples_split,
                                                              min_samples_leaf=min_samples_leaf), \
               'neg_mean_squared_error'
    elif name == "KNN_Classification":
        return KNeighborsClassifier(), dict(leaf_size=leaf_size, n_neighbors=n_neighbors), 'accuracy'
    elif name == 'Random_Forest_Classification':
        return sklearn.ensemble.RandomForestClassifier(), dict(n_estimators=n_estimators,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf), 'accuracy'
