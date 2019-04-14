from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_breast_cancer()
# feature_count = len(dataset.data[0])
feature_count = 10


def count_error(features):
    data = dataset.data[:, features]

    gnb = GaussianNB()
    gnb.fit(data, dataset.target)

    y_predict = gnb.predict(data)

    return (dataset.target != y_predict).sum()
