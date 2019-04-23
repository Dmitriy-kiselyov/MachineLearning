from feature_selection.data import count_error, get_feature_count


def selection_del(dataset, result=None):
    feature_count = get_feature_count(dataset)
    if result is None:
        features_all = list(range(feature_count))
        result = {
            "error": count_error(dataset, features_all),
            "features": features_all
        }

    while len(result["features"]) > 1:
        result_cur = result

        for feat in result["features"]:
            features = result["features"][:]
            features.remove(feat)

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
