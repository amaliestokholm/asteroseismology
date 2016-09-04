import os
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


# Make an Échelle diagram
def echelle(starname, minfreq, maxfreq):
    print('Plot Échelle diagram')
    dir = './X072495_y02638_nor_4_495_595'
    datafiles = sorted([s for s in os.listdir(dir)])
    for i, datafile in enumerate(datafiles):
        path = os.path.join(dir, datafile)
        l, n, f, a = np.loadtxt(path, usecols=(0, 1, 2, 3)).T
        plt.figure()
        delta_nu = np.median(np.diff(f[n == 0]))
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
        plt.plot(np.mod(f[l == 0], delta_nu), f[l == 0], 'bo')
        plt.plot(np.mod(f[l == 1], delta_nu), f[l == 1], 'go')
        plt.plot(np.mod(f[l == 2], delta_nu), f[l == 2], 'yo')
        plt.plot(np.mod(f[l == 3], delta_nu), f[l == 3], 'mo')
        plt.title(r'The Echelle diagram of %s with $\Delta\nu=$%s' %
                  (starname, delta_nu))
        plt.xlabel(r'Frequency mod $\Delta\nu$ [$\mu$Hz]')
        plt.ylabel(r'Frequency [$\mu$Hz]')
        plt.xlim([0, delta_nu])
        plt.savefig('./echelle/%s_echelle_%s_%s_%s.pdf' % (starname, i,
                                                           minfreq, maxfreq))


echelle('HD181096', 495, 595)
