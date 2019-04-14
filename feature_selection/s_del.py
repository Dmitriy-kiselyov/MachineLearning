from feature_selection.data import count_error, feature_count


def selection_del(result=None):
    if result is None:
        features_all = list(range(feature_count))
        result = {
            "error": count_error(features_all),
            "features": features_all
        }

    while len(result["features"]) > 1:
        result_cur = result

        for feat in result["features"]:
            features = result["features"][:]
            features.remove(feat)

            error = count_error(features)
            if error <= result_cur["error"]:
                result_cur = {
                    "error": error,
                    "features": features
                }

        if result_cur == result:
            break

        result = result_cur

    return result
