import matplotlib.pyplot as plot


def plot_dataset(dataset, colors):
    def get_by_y(y):
        return dataset.data[list(map(lambda t: t == y, dataset.target))]

    for y in colors:
        data_y = get_by_y(y)
        plot.scatter(data_y[:, 0], data_y[:, 1], edgecolor='black', color=y, s=50, zorder=100)

    __scale(dataset.data[:, 0], plot.xlim)
    __scale(dataset.data[:, 1], plot.ylim)


def __scale(data, scale_fn):
    d_min, d_max = min(data), max(data)
    diff = d_max - d_min
    delta = diff / 8

    scale_fn(d_min - delta, d_max + delta)


def plot_gain(gain, zorder):
    from_value = gain["value"]["from"]
    to_value = gain["value"]["to"]
    line_type = plot.axhspan if gain["feat"] == 1 else plot.axvspan

    if to_value == float('inf'):
        to_value = 100
    if from_value == -float('inf'):
        from_value = -100

    line_type(from_value, to_value, facecolor=gain["class"], zorder=zorder)
