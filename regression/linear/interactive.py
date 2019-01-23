import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

from regression.linear.linear import LinearRegression


def sklearn_predict(X_train, y, X_test, preprocess):
    regr = linear_model.LinearRegression(fit_intercept=preprocess)
    regr.fit(X_train, y)

    print('SKYKIT: ', np.round(regr.coef_, 2))

    return regr.predict(X_test)


def linear_predict(X_train, y, X_test, preprocess):
    regr = LinearRegression(preprocess=preprocess)
    regr.fit(X_train, y)

    print('LINEAR: ', np.round(regr.coeffs, 2))

    return regr.predict(X_test)


def svd_predict(X_train, y, X_test, preprocess):
    regr = LinearRegression(svd=True, preprocess=preprocess)
    regr.fit(X_train, y)

    print('SVD: ', np.round(regr.coeffs, 2))

    return regr.predict(X_test)


# Load the diabetes dataset
diabetes = datasets.load_diabetes()  # 442 Объекта, 10 признаков

# Use only some features
def prepare_dataset(size, features):
    diabetes_X = diabetes.data[:(size * 2), :features]
    diabetes_y = diabetes.target[:(size * 2)]

    diabetes_X_train = diabetes_X[:size]
    diabetes_X_test = diabetes_X[-size:]

    diabetes_y_train = diabetes_y[:size]
    diabetes_y_test = diabetes_y[-size:]

    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test

def compare(size, features, preprocess):
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = prepare_dataset(size, features)

    sk_y_est = sklearn_predict(diabetes_X_train, diabetes_y_train, diabetes_X_test, preprocess)
    print("Средняя квадратичная ошика: %.2f"
          % mean_squared_error(diabetes_y_test, sk_y_est))

    linear_y_est = linear_predict(diabetes_X_train, diabetes_y_train, diabetes_X_test, preprocess)
    print("Средняя квадратичная ошика: %.2f"
          % mean_squared_error(diabetes_y_test, linear_y_est))

    svd_y_est = svd_predict(diabetes_X_train, diabetes_y_train, diabetes_X_test, preprocess)
    print("Средняя квадратичная ошика: %.2f"
          % mean_squared_error(diabetes_y_test, svd_y_est))

    print('---------------------------------------------------------')
    print('Правильно: ', diabetes_y_test)
    print('Skykit:    ', np.floor(sk_y_est))
    print('Linear:    ', np.floor(linear_y_est))
    print('SVD:       ', np.floor(svd_y_est))


compare(50, 3, preprocess=True)
