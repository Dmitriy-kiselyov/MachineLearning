from matplotlib import pyplot as plot
from sklearn import datasets

from logical.decision_list import create_decision_list
from logical.data import prepare
from logical.plot import plot_dataset, plot_gain


colors = ['r', 'b', 'g']


def fetch_data():
    features = [0, 3]
    return prepare(datasets.load_iris(), features, colors)


gains = []
iris = fetch_data()


def logger(best_gain, predicate):
    print("----------------------------")
    log_best_gain(best_gain)
    print("PREDICATE:", predicate.to_str())

    gains.append(best_gain)

    print("<PLOT>")

    plot_dataset(iris, colors)
    for i in range(len(gains)):
        plot_gain(gains[i], zorder=len(gains)-i)
    plot.show()


def log_best_gain(gain):
    print("BEST: F =", gain["feat"], ", C =", gain["class"], ", GAIN =", gain["gain"], "from", gain["value"]["from"], "to", gain["value"]["to"])


plot_dataset(iris, colors)
plot.show()

predicates = create_decision_list(fetch_data(), logger=logger)

print("----------------------------")
print("Список предикатов:")
for p in predicates:
    print(p.to_str())


