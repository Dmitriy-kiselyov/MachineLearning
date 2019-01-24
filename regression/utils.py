import numpy as np
import random


def makerange(lo, hi, size):
    step = (hi - lo) / (size - 1)
    range = np.arange(lo, hi + step, step) # операция неточная, размер может быть больше
    return range if len(range) == size else range[0:size]


def makenoise(lo, hi, size):
    randomized = np.asarray(random.sample(range(size), size))
    difference = (hi - lo) / (size - 1)
    return randomized * difference + lo


def distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def make_errors(value, size, rate):
    errors = []

    for i in range(size):
        if random.random() < rate:
            errors.append(value if random.randint(0, 1) == 1 else -value)
        else:
            errors.append(0)

    return errors


def randomize_cols(matrix):
    size = len(matrix[0])
    randomized = np.asarray(random.sample(range(size), size))

    new_m = None
    for i in randomized:
        col = matrix[:, i:i+1]
        new_m = col if new_m is None else np.hstack((new_m, col))

    return new_m
