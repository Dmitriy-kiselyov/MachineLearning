from regression.utils import distance
import numpy as np


def calc_h_errors(xl, yl, hl, kernel):
    h_errors = []

    for h in hl:
        h_error = 0
        for i, x in enumerate(xl):
            y_r = nw_loo(i, xl, yl, h, kernel)
            h_error += (y_r - yl[i]) ** 2
        h_errors.append(h_error)

    return h_errors


def nw_loo_estimate(xl, yl, h, kernel):
    size = len(xl)
    return list(map(lambda i: nw_loo(i, xl, yl, h, kernel), range(size)))


def nw_loo(i, xl, yl, h, kernel):
    return nw(xl[i], np.delete(xl, i), np.delete(yl, i), h, kernel)


def nw(x, xl, yl, h, kernel):
    top, bottom = 0, 0
    for i, x_i in enumerate(xl):
        k = kernel(distance(x, x_i) / h)
        top += yl[i] * k
        bottom += k

    return float('inf') if bottom == 0 else top / bottom
