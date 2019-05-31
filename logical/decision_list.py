from logical.data import remove_from_dataset
from logical.entropy import find_best_gain
from logical.predicate import create_predicate


def create_decision_list(dataset, logger=None):
    log = logger is not None

    predicates = []

    while len(dataset.data) > 0:
        best_gain = find_best_gain(dataset, log=log)

        predicate = create_predicate(best_gain)

        remove = []
        for i in range(len(dataset.data)):
            x = dataset.data[i]
            if predicate(x):
                remove.append(i)

        if log:
            print("REMOVE", len(remove), "POINTS")

        remove_from_dataset(dataset, remove)

        if logger:
            logger(best_gain, predicate)

        predicates.append(predicate)

    return predicates
