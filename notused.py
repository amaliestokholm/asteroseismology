import numpy as np
import matplotlib


def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    fig_width_pt = 328
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('figure', figsize=fig_size)
    matplotlib.rc('font', size=8, family='serif')
    matplotlib.rc('axes', labelsize=8)
    matplotlib.rc('legend', fontsize=8)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('text.latex', preamble=
                  r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

matplotlib_setup()
import matplotlib.pyplot as plt
import plots

def second_largest(a):
    """
    >>> second_largest([3,2,1])
    1
    >>> second_largest([2,3,1])
    0
    >>> second_largest([1,3,2])
    2
    >>> second_largest([1,2,3])
    1
    >>> second_largest([1])
    0
    >>> second_largest([2, 1])
    1
    """
    a = np.asarray(a)
    if len(a) == 1:
        return 0
    i = np.argmax(a)
    left = np.argmax(a[:i]) if i > 0 else 1
    right = (i+1 + np.argmax(a[i+1:])) if i+1 < len(a) else i-1
    if a[left] > a[right]:
        return left
    else:
        return right


# NOT USED
def plot_v2():
    beta, v2, sigma_v2, u, v = np.loadtxt('mean2.txt',skiprows=1).T
    
    plt.figure()
    plt.plot(beta, v2, color='k', marker='.', ms=3, linestyle='None')
    plt.errorbar(beta, v2, sigma_v2, fmt=None, color='k')
    plt.plot(u, v, 'r', linewidth=0.5)
    plt.title(r'The Visibility Curve of %s' % starname)
    plt.xlabel(r'Spatial frequency [rad$^{-1}$]')
    plt.ylabel(r'Visibility V$^2$')
    plt.savefig('%s_visibility_%s_%s.pdf' % (starname,
                minfreq, maxfreq))
    plt.show()

# NOT USED

