from sklearn import datasets
import matplotlib.pyplot as plot

from logical.data import remove_same
from logical.compress import compress_by_class, compress_to_class_str

iris = datasets.load_iris()
colors = ['r', 'b', 'g']
f1, f2 = 0, 3

iris.data = iris.data[:, [f1, f2]]
remove_same(iris)


def show_2_features():
    def get_by_y(y):
        return iris.data[list(map(lambda t: t == y, iris.target))]

    for y in range(0, 3):
        data_y = get_by_y(y)
        plot.plot(data_y[:, 0], data_y[:, 1], 'o', color=colors[y], markersize=10)

    plot.show()


show_2_features()

features = iris.data[:, 1]
classes = list(map(lambda i: colors[i], iris.target))
bind = list(map(lambda f, c: {"feat": f, "class": c}, features, classes))
bind.sort(key=lambda b: b["feat"])

comp = compress_by_class(bind)

print(compress_to_class_str(comp))
