from sklearn import datasets
import matplotlib.pyplot as plot

from logical.data import remove_conflicts, set_main_class
from logical.compress import compress_by_class, compress_to_class_str
from logical.entropy import gain

iris = datasets.load_iris()
colors = ['r', 'b', 'g']
f1, f2 = 0, 3

iris.data = iris.data[:, [f1, f2]]
remove_conflicts(iris)


def show_2_features():
    def get_by_y(y):
        return iris.data[list(map(lambda t: t == y, iris.target))]

    for y in range(0, 3):
        data_y = get_by_y(y)
        plot.plot(data_y[:, 0], data_y[:, 1], 'o', color=colors[y], markersize=10)

    plot.show()


# show_2_features()

def get_compressed_str(i, main=None):
    classes = list(map(lambda i: colors[i], iris.target))
    if main:
        classes = set_main_class(classes, main)

    features = iris.data[:, i]
    bind = list(map(lambda f, c: {"feat": f, "class": c}, features, classes))
    bind.sort(key=lambda b: b["feat"])
    comp = compress_by_class(bind)

    return compress_to_class_str(comp)


print(get_compressed_str(1))
print(get_compressed_str(1, 'b'))

P_arr = [0, 41, 3, 1, 0]
N_arr = [50, 0, 1, 1, 44]
P = sum(P_arr)
N = sum(N_arr)

p = 0
n = 0

print("P =", P, "; N =", N)

for i in range(len(P_arr) - 1):
    p += P_arr[i]
    n += N_arr[i]

    print("p =", p, "; n =", n, " ; gain =", gain(P, N, p, n))
