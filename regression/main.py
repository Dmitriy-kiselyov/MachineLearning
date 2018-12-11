import numpy as np
import matplotlib.pyplot as plot

import regression.kernels as kernels
import regression.datasets as datasets
from regression.nw_regression import nw_regression, loo as nw_loo

size = 100
data = datasets.sin_with_noise(-2 * np.pi, 4 * np.pi, size, noise=0.3)
xl, yl = data['xl'], data['yl']

plot.plot(xl, yl)
plot.xlabel('x')
plot.ylabel('sin(x)')
plot.show()

def test_by_kernel(kernel):
    hl = np.arange(0.1, 1, 0.05)
    h_errors = nw_loo(xl, yl, hl, kernel)
    print('H errors', h_errors)

    min_i = np.argmin(h_errors)
    h_error = round(h_errors[min_i], 5)
    best_h = round(hl[min_i], 1)
    yl_r = list(map(lambda x: nw_regression(x, xl, yl, best_h, kernel), xl))

    plot.plot(xl, yl)
    plot.plot(xl, yl_r)
    plot.xlabel('x, h = ' + str(best_h))
    plot.ylabel('sin(x), mistake = ' + str(h_error))
    plot.show()

test_by_kernel(kernels.gauss)
test_by_kernel(kernels.quartic)
