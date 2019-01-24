import numpy as np
import matplotlib.pyplot as plot

import regression.kernels as kernels
import regression.datasets as datasets
from regression.nw.nw import nw_loo_estimate, calc_h_errors as nw_calc_h_errors
from regression.lowess.lowess import lowess_loo_estimate, calc_h_errors as lowess_calc_h_errors, calc_weights


def lowess_interactive(size, fn_name, stable_limit, error, error_rate):
    data = getattr(datasets, fn_name)(-3 * np.pi, 4 * np.pi, size, error, rate=error_rate)
    xl, yl = data['xl'], data['yl']

    plot.rcParams['figure.figsize'] = (15, 6)
    plot.plot(xl, yl)
    plot.xlabel('x')
    plot.ylabel(fn_name + '(x)')
    plot.show()

    plot.figure()
    plot.subplot(1, 2, 1)
    __plot_nw(xl, yl, fn_name)

    plot.subplot(1, 2, 2)
    __plot_lowess(xl, yl, fn_name, stable_limit)
    plot.show()


def __plot_nw(xl, yl, fn_name):
    kernel = kernels.quartic
    hl = np.arange(0.2, 0.7, 0.05)

    h_errors = nw_calc_h_errors(xl, yl, hl, kernel)
    min_i = np.argmin(h_errors)
    h_error = round(h_errors[min_i], 5)
    best_h = hl[min_i]

    yl_est = nw_loo_estimate(xl, yl, best_h, kernel)

    plot.plot(xl, yl)
    plot.plot(xl, yl_est)
    plot.xlabel('x, h = ' + str(round(best_h, 2)))
    plot.ylabel(fn_name + '(x), error = ' + str(h_error))
    plot.title("Непараметрическая регрессия")


def __plot_lowess(xl, yl, fn_name, stable_limit):
    hl = np.arange(0.05, 0.3, 0.05)

    h_errors = lowess_calc_h_errors(xl, yl, hl, stable_limit)
    min_i = np.argmin(h_errors)
    h_error = round(h_errors[min_i], 5)
    best_h = hl[min_i]

    weights = calc_weights(xl, yl, best_h, stable_limit)
    yl_est = lowess_loo_estimate(xl, yl, weights, best_h)

    plot.plot(xl, yl)
    plot.plot(xl, yl_est)
    plot.xlabel('x, h = ' + str(round(best_h, 2)))
    plot.ylabel(fn_name + '(x), error = ' + str(h_error))
    plot.title("Lowess")
