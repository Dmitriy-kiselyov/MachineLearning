import numpy as np


class SVD:
    def __init__(self, preprocess=True, cut=True):
        self.__has_preprocess = preprocess
        self.__cut = cut
        self.__intercept = 0

    def fit(self, X, y):
        if self.has_preprocess:
            X, y, X_offset, y_offset = self.__preprocess(X, y)

        V, D, U = self.__svd(X)

        pseudo_inverse = np.dot(np.dot(U, np.linalg.inv(D)), np.transpose(V))
        self.__w = np.dot(pseudo_inverse, y)

        if self.has_preprocess:
            self.__set_intercept(X_offset, y_offset)

    def predict(self, X):
        if self.uses_cut:
            X = X[:, :2]

        return np.dot(X, self.__w) + self.__intercept

    @property
    def coeffs(self):
        return self.__w

    @property
    def has_preprocess(self):
        return self.__has_preprocess

    @property
    def uses_cut(self):
        return self.__cut

    def __set_intercept(self, X_offset, y_offset):
        self.__intercept = y_offset - np.dot(X_offset, self.coeffs)

    def __preprocess(self, X, y):
        if self.uses_cut:
            X = X[:, :2]

        X_offset = np.average(X, axis=0)
        X = X - X_offset

        y_offset = np.average(y, axis=0)
        y = y - y_offset

        return X, y, X_offset, y_offset

    def __svd(self, X):
        V, D, U = np.linalg.svd(X, full_matrices=False)
        U = np.transpose(U)
        D = np.diag(D)

        if self.uses_cut:
            V = V[:, :2]
            D = D[:2, :2]
            U = U[:2, :2]

        return V, D, U
