import numpy as np
from sklearn import datasets
import time

from feature_selection.s_add import selection_add
from feature_selection.s_del import selection_del
from feature_selection.add_del import selection_add_del
from feature_selection.data import count_error
from feature_selection.full_search import selection_full_search
from feature_selection.dfs import selection_dfs
from feature_selection.bfs import selection_bfs


dataset = datasets.load_breast_cancer()
feature_count = 30
dataset.data = dataset.data[:, :feature_count]


def measure_time(func):
    start = time.process_time()
    result = func()
    elapsed = int((time.process_time() - start) * 1000)

    print("Время работы:", "{:,}".format(elapsed).replace(",", " "), "мс")

    return result


def log_quality(result):
    error = result["error"]
    total = len(dataset.data)
    quality = round((total - error) / total * 100, 2)

    print("Качество алгоритма:", str(quality) + "%")


def log_features(result):
    features = np.asarray(result["features"])
    names = dataset.feature_names

    print("Признаки: ", list(map(lambda i: str(i) + " – " + names[i], features)))


def log(message, func):
    print(message)

    result = measure_time(func)
    print("Количество ошибок:", result["error"])
    log_quality(result)
    print("Количество признаков: ", len(result["features"]))
    log_features(result)
    print("-------------------------------------------")


features_all = list(range(feature_count))
log("Полный набор признаков", lambda: {
    "error": count_error(dataset, features_all),
    "features": features_all
})

if feature_count <= 15:
    log("Полный перебор", lambda: selection_full_search(dataset))

log("Алгоритм ADD", lambda: selection_add(dataset))

log("Алгоритм DEL", lambda: selection_del(dataset))

log("Алгоритм ADD-DEL", lambda: selection_add_del(dataset))

log('Поиск в глубину', lambda: selection_dfs(dataset))

log('Поиск в ширину', lambda: selection_bfs(dataset, iter_limit=10))
