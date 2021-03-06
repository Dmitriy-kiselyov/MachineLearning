import numpy as np


class LinearRegression:
    def __init__(self, preprocess=True):
        self.__has_preprocess = preprocess
        self.__intercept = 0

    def fit(self, X, y):
        if self.has_preprocess:
            X, y, X_offset, y_offset = self.__preprocess(X, y)

        F = X
        F_transpose = np.transpose(F)

        pseudo_inverse = np.dot(np.linalg.inv(np.dot(F_transpose, F)), F_transpose)
        self.__w = np.dot(pseudo_inverse, y)

        if self.has_preprocess:
            self.__set_intercept(X_offset, y_offset)

    def predict(self, X):
        return np.dot(X, self.__w) + self.__intercept

    @property
    def coeffs(self):
        return self.__w

    @property
    def has_preprocess(self):
        return self.__has_preprocess

    def __pseudo_linear(self, F):
        F_transpose = np.transpose(F)

        return np.dot(np.linalg.inv(np.dot(F_transpose, F)), F_transpose)

    def __set_intercept(self, X_offset, y_offset):
        self.__intercept = y_offset - np.dot(X_offset, self.coeffs)

    def __preprocess(self, X, y):
        X_offset = np.average(X, axis=0)
        X = X - X_offset

        y_offset = np.average(y, axis=0)
        y = y - y_offset

        return X, y, X_offset, y_offset
