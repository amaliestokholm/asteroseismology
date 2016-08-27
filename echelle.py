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
def echelle(delta_nu):
    print('Plot Échelle diagram')
    dir = './X072495_y02638_nor_4_495_595'
    datafiles = sorted([s for s in os.listdir(dir)])
    for datafile in datafiles:
        path = os.path.join(dir, datafile)
        n, l, f, a = np.loadtxt(path, usecols=(0,1,2,3)).T
    
    """
    plt.figure()
    plt.scatter(peakmod, peak, c=height, cmap='gray')
    #plt.plot(np.mod(timpeak, delta_nu), timpeak, 'bo')
    plt.plot(np.mod(f, delta_nu), f, 'ro')
    #plt.plot(np.mod(amaliepeak, delta_nu), amaliepeak,  'ro')
    plt.title(r'The Echelle diagram of %s with $\Delta\nu=$%s' %
              (starname, delta_nu))
    plt.xlabel(r'Frequency mod $\Delta\nu$ [$\mu$Hz]')
    plt.ylabel(r'Frequency [$\mu$Hz]')
    plt.xlim([0, delta_nu])
    plt.savefig('%s_echelle_%s_%s.pdf' % (starname, minfreq, maxfreq))
    plt.show()
    return peak, height
    """

echelle(53.7)
