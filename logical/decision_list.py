from sklearn import datasets
import matplotlib.pyplot as plot

from logical.data import remove_conflicts, remove_from_dataset
from logical.entropy import find_best_gain
from logical.predicate import create_predicate

iris = datasets.load_iris()
colors = ['r', 'b', 'g']
f1, f2 = 0, 3

iris.data = iris.data[:, [f1, f2]]
remove_conflicts(iris)
iris.target = list(map(lambda t: colors[t], iris.target))


def show_2_features():
    def get_by_y(y):
        return iris.data[list(map(lambda t: t == y, iris.target))]

    for y in colors:
        data_y = get_by_y(y)
        plot.plot(data_y[:, 0], data_y[:, 1], 'o', color=y, markersize=10)

    plot.show()


show_2_features()

while len(iris.data) > 0:
    best_gain = find_best_gain(iris, log=True)

    print("BEST", best_gain)

    predicate = create_predicate(best_gain)

    remove = []
    for i in range(len(iris.data)):
        x = iris.data[i]
        if predicate(x):
            remove.append(i)

    print("REMOVE", remove)
    remove_from_dataset(iris, remove)

    print("-------------------")
