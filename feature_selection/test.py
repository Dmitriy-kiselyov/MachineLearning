import numpy as np
from sklearn import datasets

from feature_selection.s_add import selection_add
from feature_selection.s_del import selection_del
from feature_selection.add_del import selection_add_del
from feature_selection.data import count_error
from feature_selection.full_search import selection_full_search
from feature_selection.dfs import selection_dfs


dataset = datasets.load_breast_cancer()
feature_count = 30
dataset.data = dataset.data[:, :feature_count]


def log(message, result):
    print(message)
    print("Количество ошибок: ", result["error"])
    print("Количество признаков: ", len(result["features"]))
    print("Признаки: ", np.asarray(result["features"]))
    print("-------------------------------------------")


features_all = list(range(feature_count))
log("Полный набор признаков", {
    "error": count_error(dataset, features_all),
    "features": features_all
})

if feature_count <= 15:
    log("Полный перебор", selection_full_search(dataset))

log("Алгоритм ADD", selection_add(dataset))

log("Алгоритм DEL", selection_del(dataset))

log("Алгоритм ADD-DEL", selection_add_del(dataset))

log('Поиск в глубину', selection_dfs(dataset))
