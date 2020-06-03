from sklearn.model_selection import train_test_split
import random


def train_test_splitting(x, y, test_size=0.3):
    random.seed(555)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=555)
    return x_train, x_test, y_train, y_test
