from feature_selection.data import count_error, get_feature_count


def selection_del(dataset, result=None):
    feature_count = get_feature_count(dataset)
    log = []
    if result is None:
        features_all = list(range(feature_count))
        error = count_error(dataset, features_all)

        log.append({
            "error": error,
            "feature_count": feature_count
        })

        result = {
            "error": error,
            "features": features_all,
            "log": log
        }

    while len(result["features"]) > 1:
        result_cur = result

        for feat in result["features"]:
            features = result["features"][:]
            features.remove(feat)

            error = count_error(dataset, features)

            log.append({
                "error": error,
                "feature_count": len(features)
            })

            if error <= result_cur["error"]:
                result_cur = {
                    "error": error,
                    "features": features,
                    "log": log
                }

        if result_cur == result:
            break

        result = result_cur

    return result
