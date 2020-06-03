import os
import pickle
from pathlib import Path
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, accuracy_score

path_a = Path(os.getcwd()).parent
save_path = os.path.join(path_a, f'data\output')
if not os.path.exists(save_path):
    path_a = Path(os.getcwd())


def save_model(model, name):
    path = os.path.join(path_a, f'data\output\{name}.pkl')
    pickle.dump(model, open(path, 'wb'))


def save_performance_regression(performance, name, y_test, y_predict, lapsed_time, fraction=1):
    # Calculate MSE
    mse = mean_squared_error(y_test, y_predict)
    # Calculate confidence interval
    confidence_interval = pd.DataFrame((y_test - y_predict) ** 2)
    stats = confidence_interval.agg(['mean', 'count', 'std'])
    ci95_hi = (stats.loc['mean'] + 1.96 * stats.loc['std'] / math.sqrt(stats.loc['count'])).values[0]
    ci95_lo = (stats.loc['mean'] - 1.96 * stats.loc['std'] / math.sqrt(stats.loc['count'])).values[0]
    # Add information to the performance data frame
    if fraction != 1:
        model_name = name + '(' + str(fraction * 100) + '%)'
    else:
        model_name = name
    performance = performance[performance.Name != model_name]  # Avoid duplicates
    df = pd.DataFrame(
        {'Name': [model_name], 'MSE': [int(mse)], 'CI_low': [int(ci95_lo)], 'CI_high': [int(ci95_hi)],
         'Time': [lapsed_time]})
    performance = performance.append(df, ignore_index=True)
    performance = performance.sort_values(by=['MSE'])
    performance.reset_index(drop=True, inplace=True)
    # Save performance data frame
    path = os.path.join(path_a, f'data\output\performance_regression.csv')
    performance.to_csv(path)
    print(performance)


def save_performance_classification(performance, name, y_test, y_predict, lapsed_time, fraction=1):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)
    # Calculate confidence interval
    stats = y_test.agg(['count'])
    ci95_hi = accuracy + 1.96 * math.sqrt((accuracy * (1 - accuracy)) / stats.loc['count'])
    ci95_lo = accuracy - 1.96 * math.sqrt((accuracy * (1 - accuracy)) / stats.loc['count'])
    # Add information to the performance data frame
    if fraction != 1:
        model_name = name + '(' + str(fraction * 100) + '%)'
    else:
        model_name = name
    performance = performance[performance.Name != model_name]  # Avoid duplicates
    df = pd.DataFrame(
        {'Name': [model_name], 'Accuracy': [accuracy], 'CI_low': [ci95_lo], 'CI_high': [ci95_hi],
         'Time': [lapsed_time]})
    performance = performance.append(df, ignore_index=True)
    performance = performance.sort_values(by=['Accuracy'], ascending=False)
    performance.reset_index(drop=True, inplace=True)
    # Save performance data frame
    path = os.path.join(path_a, f'data\output\performance_classification.csv')
    performance.to_csv(path)
    print(performance)
