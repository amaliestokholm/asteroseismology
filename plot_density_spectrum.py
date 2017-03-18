import numpy as np
import matplotlib.pyplot as plt


def binavg(x, bins):
    sz = len(x) // bins
    x = x[:sz * bins]
    x = x.reshape(-1, sz)
    x = np.mean(x, axis=1)
    return x

def logbins(x, y, bins):
    gt = (x > 0)
    x = x[gt]
    y = y[gt]
    d = np.logspace(np.log10(x[0]), np.log10(x[-1]), bins + 1)
    which_bin = np.searchsorted(d, x)
    bin_indexes = np.unique(which_bin)
    means = []
    for v in bin_indexes:
        means.append(np.mean(y[which_bin == v]))
    return np.asarray(means)

def loglog_plot(x, y):
    fig, ax = plt.subplots()
    ax.loglog()
    ax.plot(x, y)
    return fig


data = np.loadtxt("/home/amalie/Dropbox/Uddannelse/UNI/1516 - fysik 3. år/Bachelorprojekt/asteroseismology/data/181096/powerseries/q1_s4k79c100.txt")
x, y = data.T
z = y / x
loglog_plot(logbins(x, x, 1000), logbins(x, z, 1000))
plt.show()
