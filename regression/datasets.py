import numpy as np

from regression.utils import makerange, makenoise, make_errors


def sin_with_noise(lo, hi, size, noise):
    return with_noise(np.sin, lo, hi, size, noise)


def sin_with_error(lo, hi, size, error, rate):
    return with_error(np.sin, lo, hi, size, error, rate)


def cos_with_noise(lo, hi, size, noise):
    return with_noise(np.cos, lo, hi, size, noise)


def cos_with_error(lo, hi, size, error, rate):
    return with_error(np.cos, lo, hi, size, error, rate)


def ox_with_noise(lo, hi, size, noise):
    return with_noise(lambda x: 0, lo, hi, size, noise)


def ox_with_error(lo, hi, size, error, rate):
    return with_error(lambda x: 0, lo, hi, size, error, rate)


def with_noise(fn, lo, hi, size, noise):
    xl = makerange(lo, hi, size)
    yl = fn(xl) + makenoise(-noise, noise, size)
    return {'xl': xl, 'yl': yl}


def with_error(fn, lo, hi, size, value, rate):
    xl = makerange(lo, hi, size)
    yl = fn(xl) + make_errors(value, size, rate)
    return {'xl': xl, 'yl': yl}


