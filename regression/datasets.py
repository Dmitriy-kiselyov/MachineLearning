import numpy as np

from regression.utils import makerange, makenoise


def sin_with_noise(lo, hi, size, noise=0.2):
    xl = makerange(lo, hi, size)
    yl = np.sin(xl) + makenoise(-noise, noise, size)
    return {'xl': xl, 'yl': yl}
