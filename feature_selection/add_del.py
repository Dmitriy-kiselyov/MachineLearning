from feature_selection.s_add import selection_add
from feature_selection.s_del import selection_del


def selection_add_del(dataset):
    log = []
    result = {
        "error": 9999,
        "features": [],
        "log": log
    }

    while True:
        new_result = selection_add(dataset, result)
        new_result = selection_del(dataset, new_result)

        if new_result["error"] >= result["error"]:
            break

        result = new_result

    return result
