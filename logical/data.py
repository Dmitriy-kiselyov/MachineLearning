import numpy as np


def remove_same(dataset):
    data, target = dataset.data, dataset.target

    remove = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (data[i] == data[j]).all() and target[i] != target[j]:
                remove.append(i)
                remove.append(j)

    dataset.data = np.delete(data, remove, axis=0)
    dataset.target = np.delete(target, remove)