import numpy as np

from regression.utils import makerange, makenoise


def sin_with_noise(lo, hi, size, noise=0.2):
    return with_noise(np.sin, lo, hi, size, noise)

def cos_with_noise(lo, hi, size, noise=0.2):
    return with_noise(np.cos, lo, hi, size, noise)

def ox_with_noise(lo, hi, size, noise=0.2):
    return with_noise(lambda x: 0, lo, hi, size, noise)

def with_noise(fn, lo, hi, size, noise=0.2):
    xl = makerange(lo, hi, size)
    yl = fn(xl) + makenoise(-noise, noise, size)
    return {'xl': xl, 'yl': yl}
