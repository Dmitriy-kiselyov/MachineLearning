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

def get_compressed(feat, main=None):
    classes = list(map(lambda i: colors[i], iris.target))
    if main:
        classes = set_main_class(classes, main)

    features = iris.data[:, feat]
    bind = list(map(lambda f, c: {"feat": f, "class": c}, features, classes))
    bind.sort(key=lambda b: b["feat"])

    return compress_by_class(bind)


def get_compressed_str(feat, main=None):
    return compress_to_class_str(get_compressed(feat, main))


best_gain = {
    "gain": 0,
    "feat": -1,
    "value": 0,
    "class": "✘"
}

def find_best_gain(feat, main):
    global best_gain

    main_i = colors.index(main)
    P = sum(iris.target == main_i)
    N = len(iris.target) - P

    print(get_compressed_str(feat, main))

    comp = get_compressed(feat, main)
    p, n = 0, 0

    for i in range(len(comp) - 1):
        p += comp[i]["class"].get("✔", 0)
        n += comp[i]["class"].get("✘", 0)

        g = gain(P, N, p, n)

        if g > best_gain["gain"]:
            best_gain = {
                "gain": g,
                "feat": feat,
                "value": comp[i]["feat_to"],
                "class": main
            }

        # print("p =", p, " n =", n, " gain =", gain(P, N, p, n))


for feat in range(2):
    for color in colors:
        find_best_gain(feat, color)

print("BEST", best_gain)
