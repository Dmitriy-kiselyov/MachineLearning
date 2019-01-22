import numpy as np
import matplotlib.pyplot as plot
plot.rcParams['figure.figsize'] = (15, 6)

import regression.kernels as kernels
import regression.datasets as datasets
from regression.nw.nw import nw_loo_estimate, calc_h_errors


def nw_interactive(size, fn_name, noise):
    data = getattr(datasets, fn_name)(-2 * np.pi, 4 * np.pi, size, noise)
    xl, yl = data['xl'], data['yl']

    plot.plot(xl, yl)
    plot.xlabel('x')
    plot.ylabel(fn_name + '(x)')
    plot.show()

    def test_by_kernel(kernel, kernel_name, hl):
        h_errors = calc_h_errors(xl, yl, hl, kernel)

        min_i = np.argmin(h_errors)
        h_error = round(h_errors[min_i], 5)
        best_h = round(hl[min_i], 1)
        yl_est = nw_loo_estimate(xl, yl, best_h, kernel)

        plot.plot(xl, yl)
        plot.plot(xl, yl_est)
        plot.xlabel('x, h = ' + str(best_h))
        plot.ylabel(fn_name + '(x), error = ' + str(h_error))
        plot.title(kernel_name)

        return h_errors

    hl = np.arange(0.1, 1.5, 0.05)

    plot.figure()
    plot.subplot(1, 2, 1)
    h_gauss_errors = test_by_kernel(kernels.gauss, 'Gauss kernel', hl)

    plot.subplot(1, 2, 2)
    h_quartic_errors = test_by_kernel(kernels.quartic, 'Quartic kernel', hl)
    plot.show()

    plot.plot(hl, h_gauss_errors)
    plot.plot(hl, h_quartic_errors)
    plot.legend(['gauss', 'quartic'])
    plot.xlabel('h')
    plot.ylabel('SSE')
    plot.show()
