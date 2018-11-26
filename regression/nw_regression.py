from regression.utils import distance

def nw_regression(x, xl, y, h, kernel):
    top, bottom = 0, 0
    for i, x_i in enumerate(xl):
        k = kernel(distance(x, x_i) / h)
        top += y[i] * k
        bottom += k

    return top / bottom