from sklearn import datasets
import matplotlib.pyplot as plot

from logical.data import remove_conflicts, remove_from_dataset
from logical.entropy import find_best_gain
from logical.predicate import create_predicate


colors = ['r', 'b', 'g']


def fetch_data():
    iris = datasets.load_iris()
    f1, f2 = 0, 3

    iris.data = iris.data[:, [f1, f2]]
    remove_conflicts(iris)
    iris.target = list(map(lambda t: colors[t], iris.target))

    return iris


iris = fetch_data()


def show_2_features():
    def get_by_y(y):
        return iris.data[list(map(lambda t: t == y, iris.target))]

    for y in colors:
        data_y = get_by_y(y)
        plot.scatter(data_y[:, 0], data_y[:, 1], edgecolor='black', color=y, s=50, zorder=100)

    def scale(data, scale_fn):
        d_min, d_max = min(data), max(data)
        diff = d_max - d_min
        delta = diff / 8

        scale_fn(d_min - delta, d_max + delta)

    scale(iris.data[:, 0], plot.xlim)
    scale(iris.data[:, 1], plot.ylim)


gains = []
predicates = []

iris_clone = fetch_data()

while len(iris_clone.data) > 0:
    best_gain = find_best_gain(iris_clone, log=True)

    print("BEST", best_gain)

    predicate = create_predicate(best_gain)

    remove = []
    for i in range(len(iris_clone.data)):
        x = iris_clone.data[i]
        if predicate(x):
            remove.append(i)

    print("REMOVE", len(remove), "POINTS")
    remove_from_dataset(iris_clone, remove)

    gains.append(best_gain)
    predicates.append(predicate)

    print("-------------------")

print(gains)


def draw_gain(gain, order):
    from_value = gain["value"]["from"]
    to_value = gain["value"]["to"]
    line_type = plot.axhspan if gain["feat"] == 1 else plot.axvspan

    if to_value == float('inf'):
        to_value = 100
    if from_value == -float('inf'):
        from_value = -100

    line_type(from_value, to_value, facecolor=gain["class"], alpha=1, zorder=order)


for n in range(1, len(gains) + 1):
    show_2_features()

    for i in range(n):
        draw_gain(gains[i], order=n-i)

    plot.show()
