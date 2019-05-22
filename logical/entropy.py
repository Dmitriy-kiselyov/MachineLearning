from math import log2

from logical.data import set_main_class
from logical.compress import compress_by_class, compress_to_class_str
from logical.data import get_feature_count


def gain(P, N, p, n):
    return __sample_entropy(P, N) - __sample_entropy_after(P, N, p, n)


def __sample_entropy_after(P, N, p, n):
    return (p + n) / (P + N) * __sample_entropy(p, n) + (P - p + N - n) / (P + N) * __sample_entropy(P - p, N - n)


def __sample_entropy(P, N):
    return 0 if P + N == 0 else __entropy(P / (P + N), N / (P + N))


def __entropy(q0, q1):
    return -q0 * __log2(q0) - q1 * __log2(q1)


def __log2(q):
    return 0 if q == 0 else log2(q)


def find_best_gain(dataset, log=False):
    best_gain = {
        "gain": 0,
        "feat": -1,
        "value": {
            "from": None,
            "to": None
        },
        "class": "✘"
    }

    for feat in range(get_feature_count(dataset)):
        for main in set(dataset.target):
            P = sum(list([t == main for t in dataset.target]))
            N = len(dataset.target) - P

            comp = __get_compressed(dataset, feat, main)
            p, n = 0, 0

            if log:
                print("F =", feat, ", C =", main, ";", compress_to_class_str(comp))

            for i in range(len(comp) - 1):
                p += comp[i]["class"].get("✔", 0)
                n += comp[i]["class"].get("✘", 0)

                g = gain(P, N, p, n)

                if g > best_gain["gain"]:
                    best_gain = {
                        "gain": g,
                        "feat": feat,
                        "value": {
                            "from": comp[i]["feat_from"],
                            "to": comp[i]["feat_to"],
                        },
                        "class": main
                    }

    return best_gain


def __get_compressed(dataset, feat, main=None):
    classes = dataset.target
    if main:
        classes = set_main_class(classes, main)

    features = dataset.data[:, feat]
    bind = list(map(lambda f, c: {"feat": f, "class": c}, features, classes))
    bind.sort(key=lambda b: b["feat"])

    return compress_by_class(bind)
