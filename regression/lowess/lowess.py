from regression.utils import distance
import numpy as np
from regression.kernels import quartic as quartic_kernel, gauss as gauss_kernel


def calc_h_errors(xl, yl, hl, stable_limit):
    h_errors = []

    for h in hl:
        weights = calc_weights(xl, yl, h, stable_limit)

        h_error = 0
        for i, x in enumerate(xl):
            y_r = lowess_loo(i, xl, yl, weights, h)
            h_error += (y_r - yl[i]) ** 2
        h_errors.append(h_error)

    return h_errors


def calc_weights(xl, yl, h, stable_limit):
    size = len(xl)
    w = np.ones(size)

    for iteration in range(50):  # Чтобы не итерироваться бесконечно в случае несходимости
        yl_est = lowess_loo_estimate(xl, yl, w, h)

        w_next = __recalc_w(np.abs(yl - yl_est))
        if __is_stable(w, w_next, stable_limit):
            break

        w = w_next

    return w


def lowess_loo_estimate(xl, yl, w, h):
    size = len(xl)
    return list(map(lambda i: lowess_loo(i, xl, yl, w, h), range(size)))


def lowess_loo(i, xl, yl, w, h):
    return lowess(xl[i], np.delete(xl, i), np.delete(yl, i), np.delete(w, i), h)


def lowess(x, xl, yl, w, h):
    top, bottom = 0, 0
    for i, x_i in enumerate(xl):
        k = gauss_kernel(distance(x, x_i) / h)
        top += yl[i] * k * w[i]
        bottom += k * w[i]

    return float('inf') if bottom == 0 else top / bottom


def __recalc_w(errors):
    median = np.median(errors)

    seq = map(lambda error: 0 if error == float('inf') else quartic_kernel(error / (6 * median)), errors)
    return np.fromiter(seq, dtype=np.double)


def __is_stable(w1, w2, eps):
    diff = np.abs(w1 - w2)

    for d in diff:
        if d > eps:
            return False
    return True

