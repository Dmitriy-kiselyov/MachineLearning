import math

def gauss(r):
    return (2 * math.pi) ** -0.5 * math.exp(-0.5 * r ** 2)

def quartic(r):
    return 15/16 * (1 - r ** 2) ** 2 * _sign(math.fabs(r) <= 1)

def _sign(condition):
    return 1 if condition else 0
