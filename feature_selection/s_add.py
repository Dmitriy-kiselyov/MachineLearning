from feature_selection.data import count_error, feature_count


def selection_add():
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
