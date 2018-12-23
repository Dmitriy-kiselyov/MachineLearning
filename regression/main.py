import numpy as np
import matplotlib.pyplot as plot

import regression.kernels as kernels
import regression.datasets as datasets
from regression.nw_regression import nw_regression, loo as nw_loo

def test_nw_regression(size, fnName, noise):
    data = getattr(datasets, fnName)(-2 * np.pi, 4 * np.pi, size, noise)
    xl, yl = data['xl'], data['yl']

    plot.plot(xl, yl)
    plot.xlabel('x')
    plot.ylabel(fnName + '(x)')
    plot.show()

    def test_by_kernel(kernel, kernelName, hl):
        h_errors = nw_loo(xl, yl, hl, kernel)

        min_i = np.argmin(h_errors)
        h_error = round(h_errors[min_i], 5)
        best_h = round(hl[min_i], 1)
        yl_r = list(map(lambda x: nw_regression(x, xl, yl, best_h, kernel), xl))

        plot.plot(xl, yl)
        plot.plot(xl, yl_r)
        plot.xlabel('x, h = ' + str(best_h))
        plot.ylabel(fnName + '(x), error = ' + str(h_error))
        plot.title(kernelName)

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
