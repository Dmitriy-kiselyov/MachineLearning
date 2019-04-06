from itertools import combinations

from feature_selection.data import count_error, feature_count


def feat_gen(count):
    for c in range(1, count + 1):
        arr = list(range(count))
        comb = combinations(arr, c)

        yield from comb


result = []

for feat_cur in feat_gen(feature_count):
    error = count_error(feat_cur)
    result.append({
        "error": error,
        "features": feat_cur
    })

result.sort(key=lambda e: e["error"])

for res in result:
    print("Features: ", res["features"])
    print("Error: ", res["error"])
    print("----------------------------")
