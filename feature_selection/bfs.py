from feature_selection.data import count_error, get_feature_count


def selection_bfs(dataset, iter_limit=1):
    feature_count = get_feature_count(dataset)
    result_best = {
        "error": 9999,
        "features": []
    }

    result = list(map(lambda i: {"features": [i]}, range(feature_count)))

    for iteration in range(1, feature_count + 1):
        for res in result:
            res["error"] = count_error(dataset, res["features"])

        result.sort(key=lambda r: r["error"])

        if len(result) > iter_limit:
            result = result[:iter_limit]

        if result[0]["error"] > result_best["error"]:
            break

        if result[0]["error"] < result_best["error"]:
            result_best = result[0]

        result_new = []
        for res in result:
            for i in range(feature_count):
                if i not in res["features"]:
                    result_new.append({"features": res["features"] + [i]})

        result = result_new

    result_best["features"].sort()

    return result_best
