from sklearn.naive_bayes import GaussianNB


def get_feature_count(dataset):
    return len(dataset.data[0])


def count_error(dataset, features):
    data = dataset.data[:, features]

    gnb = GaussianNB()
    gnb.fit(data, dataset.target)

    y_predict = gnb.predict(data)

    return (dataset.target != y_predict).sum()
