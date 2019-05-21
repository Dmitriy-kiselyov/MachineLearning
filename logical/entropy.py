from math import log2


def gain(P, N, p, n):
    return __sample_entropy(P, N) - __sample_entropy_after(P, N, p, n)


def __sample_entropy_after(P, N, p, n):
    return (p + n) / (P + N) * __sample_entropy(p, n) + (P - p + N - n) / (P + N) * __sample_entropy(P - p, N - n)


def __sample_entropy(P, N):
    return 0 if P + N == 0 else __entropy(P / (P + N), N / (P + N))


def __entropy(q0, q1):
    return -q0 * __log2(q0) - q1 * __log2(q1)


def __log2(q):
    return 0 if q == 0 else log2(q)
