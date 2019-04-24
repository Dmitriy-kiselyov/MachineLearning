from feature_selection.data import count_error, get_feature_count


def selection_dfs(dataset):
    feature_count = get_feature_count(dataset)
    features = __sort_features(dataset)

    results = list(map(lambda x: {"error": 9999, "features": []}, range(feature_count + 1)))

    __increase([], features, results, dataset)

    result = min(results, key=lambda x: x["error"])
    result["features"] = sorted(__flatten_features(result["features"]))

    return result


def __sort_features(dataset):
    feature_count = get_feature_count(dataset)
    feature_errors = list(map(lambda i: {"i": i, "error": count_error(dataset, [i])}, range(feature_count)))
    features_sorted = list(sorted(feature_errors, key=lambda x: x["error"]))
    return list(map(lambda f, pos: {"i": f["i"], "pos": pos}, features_sorted, range(feature_count)))


def __increase(features_cur, features, results, dataset):
    length = len(features_cur)
    result = results[length]
    error = count_error(dataset, __flatten_features(features_cur)) if length != 0 else 9999

    for j in range(length):
        if results[j]["error"] < error:
            return

    if error < result["error"]:
        result["error"] = error
        result["features"] = features_cur

    max_feature = __max_feature_pos(features_cur)
    for feature in features:
        if feature["pos"] > max_feature:
            __increase(features_cur + [feature], features, results, dataset)


def __flatten_features(features):
    return list(map(lambda f: f["i"], features))


def __max_feature_pos(features):
    return max(features, key=lambda f: f["pos"])["pos"] if len(features) != 0 else -1
