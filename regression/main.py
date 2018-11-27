import matplotlib.pyplot as plot
import regression.kernels as kernels
from regression.utils import makerange, makenoise
from regression.nw_regression import nw_regression, loo as nw_loo
import numpy as np

size = 100
xl = makerange(-2 * np.pi, 4 * np.pi, size)
yl = np.sin(xl) + makenoise(-0.2, 0.2, size)

plot.plot(xl, yl)
plot.xlabel('x')
plot.ylabel('sin(x)')
plot.show()

hl = np.arange(0.1, 1, 0.1)
kernel = kernels.quartic

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
