import numpy as np

from feature_selection.s_add import selection_add
from feature_selection.s_del import selection_del
from feature_selection.data import count_error, feature_count


def log(message, result):
    print(message)
    print("Количество ошибок: ", result["error"])
    print("Количество признаков: ", len(result["features"]))
    print("Признаки: ", np.asarray(result["features"]))
    print("-------------------------------------------")


features_all = list(range(feature_count))
log("Полный набор признаков", {
    "error": count_error(features_all),
    "features": features_all
})

log("Алгоритм ADD", selection_add())

log("Алгоритм DEL", selection_del())
