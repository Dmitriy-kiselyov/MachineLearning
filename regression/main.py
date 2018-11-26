import matplotlib.pyplot as plot
import regression.kernels as kernels
from regression.utils import makerange, makenoise
from regression.nw_regression import nw_regression
import numpy as np

size = 200
xl = makerange(-2 * np.pi, 4 * np.pi, size)
yl = np.sin(xl) + makenoise(-0.2, 0.2, size)

plot.plot(xl, yl)
plot.xlabel('x')
plot.ylabel('sin(x)')
plot.show()

h = 0.5
yl_r1 = list(map(lambda x: nw_regression(x, xl, yl, h, kernels.gauss), xl))
yl_r2 = list(map(lambda x: nw_regression(x, xl, yl, h, kernels.quartic), xl))

plot.plot(xl, yl)
plot.plot(xl, yl_r1) # orange
plot.plot(xl, yl_r2) # green
plot.xlabel('x')
plot.ylabel('sin(x)')
plot.show()
