from sklearn import datasets
import matplotlib.pyplot as plot

from logical.data import remove_conflicts
from logical.entropy import find_best_gain

iris = datasets.load_iris()
colors = ['r', 'b', 'g']
f1, f2 = 0, 3

iris.data = iris.data[:, [f1, f2]]
remove_conflicts(iris)
iris.target = list(map(lambda t: colors[t], iris.target))


def show_2_features():
    def get_by_y(y):
        return iris.data[list(map(lambda t: t == y, iris.target))]

    for y in range(0, 3):
        data_y = get_by_y(y)
        plot.plot(data_y[:, 0], data_y[:, 1], 'o', color=colors[y], markersize=10)

    plot.show()


# show_2_features()

print("BEST", find_best_gain(iris, log=True))
