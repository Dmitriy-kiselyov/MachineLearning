import matplotlib.pyplot as plot
import math

from feature_selection.data import get_feature_count


def make_plot(dataset, result):
    logs = result["log"]
    if not logs:
        return

    feature_count = get_feature_count(dataset)

    __set_labels()

    __plot_all(logs, feature_count)
    __plot_best_in_col(logs, feature_count)
    __plot_best(result)


def make_compare_plot(dataset, result, draw_ticks=False, **kwargs):
    logs = result["log"]
    if not logs:
        return

    feature_count = get_feature_count(dataset)

    __set_labels()

    if draw_ticks:
        __draw_x_ticks(feature_count)

    __plot_best_in_col(logs, feature_count, **kwargs)
    __plot_best(result, **kwargs)


def __set_labels():
    plot.xlabel("Количество признаков")
    plot.ylabel("Ошибка выборки")


def __draw_x_ticks(feature_count):
    xl = range(1, feature_count + 1)

    plot.xticks(xl)

    for x in xl:
        plot.axvline(x, linestyle='--', linewidth=1)


def __plot_all(logs, feature_count):
    xl = list(map(lambda log: log["feature_count"], logs))
    yl = list(map(lambda log: log["error"], logs))

    plot.xticks(range(1, feature_count + 1))
    plot.yticks(__make_y_ticks(logs))

    plot.plot(xl, yl, 'o', markerfacecolor='none')


def __plot_best_in_col(logs, feature_count, linestyle='-', color='red', linewidth=2):
    xl = range(1, feature_count + 1)
    yl = [None] * feature_count

    for log in logs:
        error = log["error"]
        feature_count = log["feature_count"]

        if yl[feature_count - 1] is None or yl[feature_count - 1] > error:
            yl[feature_count - 1] = error

    plot.plot(xl, yl, 'o', color=color)
    plot.plot(xl, yl, linestyle=linestyle, color=color, linewidth=linewidth)


def __plot_best(result, color='red', **kwargs):
    x = len(result["features"])
    y = result["error"]

    plot.plot(x, y, 'o', markeredgecolor='black', markersize=10, color=color)


def __make_y_ticks(logs, height=480-110, tick_height=10):
    yl = sorted(list(set(map(lambda log: log["error"], logs))))
    size = max(yl) - min(yl) + 1
    px_for_each = height / size
    min_gap = math.ceil(tick_height / px_for_each)

    gapped_yl = [yl[0]]

    for y in yl:
        if y - gapped_yl[-1] > min_gap:
            gapped_yl.append(y)

    return gapped_yl
