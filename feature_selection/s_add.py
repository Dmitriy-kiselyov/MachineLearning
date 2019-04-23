from feature_selection.data import count_error, get_feature_count


def selection_add(dataset, result=None):
    feature_count = get_feature_count(dataset)
    if result is None:
        result = {
            "error": 9999,
            "features": []
        }

    while True:
        result_cur = result

        for feat in range(feature_count):
            if feat in result["features"]:
                continue

            features = result["features"] + [feat]
            features.sort()

            error = count_error(dataset, features)
            if error <= result_cur["error"]:
                result_cur = {
                    "error": error,
                    "features": features
                }

        if result_cur == result:
            break

        result = result_cur

    return result
