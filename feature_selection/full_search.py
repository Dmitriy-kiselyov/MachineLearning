from itertools import combinations

from feature_selection.data import count_error, feature_count


def __feat_gen(count):
    for c in range(1, count + 1):
        arr = list(range(count))
        comb = combinations(arr, c)

        yield from comb


def selection_full_search():
    result = {
        "error": 9999,
        "features": []
    }

    for feat_cur in __feat_gen(feature_count):
        error = count_error(feat_cur)

        if error < result["error"]:
            result = {
                "error": error,
                "features": feat_cur
            }

    return result
