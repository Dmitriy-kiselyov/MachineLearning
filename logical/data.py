import numpy as np


def remove_conflicts(dataset):
    data, target = dataset.data, dataset.target

    remove = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (data[i] == data[j]).all() and target[i] != target[j]:
                remove.append(i)
                remove.append(j)

    remove_from_dataset(dataset, remove)


def remove_from_dataset(dataset, remove_list):
    dataset.data = np.delete(dataset.data, remove_list, axis=0)
    dataset.target = np.delete(dataset.target, remove_list)


def set_main_class(target, main):
    return list(map(lambda t: "✔" if t == main else "✘", target))


def get_feature_count(dataset):
    return len(dataset.data[0])
