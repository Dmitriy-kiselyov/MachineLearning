import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error

from regression.utils import randomize_cols
from regression.linear.linear import LinearRegression
from regression.linear.svd import SVD


def predict(algo, X_train, y, X_test, preprocess, cut):
    regr = None
    if algo == 'linear':
        regr = LinearRegression(preprocess=preprocess)
    elif algo == 'svd':
        regr = SVD(preprocess=preprocess, cut=cut)

    regr.fit(X_train, y)

    print('Coeffs: ', np.round(regr.coeffs, 2))

    return regr.predict(X_test)


# Load the diabetes dataset
diabetes = datasets.load_diabetes()  # 442 Объекта, 10 признаков
data = randomize_cols(diabetes.data)

# Use only some features
def prepare_dataset(size, features):
    diabetes_X = data[:(size * 2), :features]
    diabetes_y = diabetes.target[:(size * 2)]

    diabetes_X_train = diabetes_X[:size]
    diabetes_X_test = diabetes_X[-size:]

    diabetes_y_train = diabetes_y[:size]
    diabetes_y_test = diabetes_y[-size:]

    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test


def test_algo(algo, size, features, preprocess, cut=False):
    print(algo, ': features =', features, '; preprocess =', preprocess)
    if cut:
        print('Using cut')

    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = prepare_dataset(size, features)

    y_est = predict(algo, diabetes_X_train, diabetes_y_train, diabetes_X_test, preprocess, cut)
    print("Средняя квадратичная ошика: %.2f"
          % mean_squared_error(diabetes_y_test, y_est))
    print('-----------------------------------------------------')


test_algo('linear', size=50, features=2, preprocess=True)
test_algo('svd', size=50, features=2, preprocess=True)
test_algo('linear', size=50, features=10, preprocess=True)
test_algo('svd', size=50, features=10, preprocess=True)
test_algo('svd', size=50, features=10, preprocess=True, cut=True)
