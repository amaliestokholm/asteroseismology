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
def echelle(starname, background=1):
    print('Plot Échelle diagram')
    dir = './X072669_Y02628_no'
    freq, power = np.loadtxt('181096.txt').T

    # Compute resolution and make dnu a multiple of the resolution
    fres = (freq[-1] - freq[0]) / (len(freq)-1)

    datafiles = sorted([s for s in os.listdir(dir) if s.startswith('obs')])
    for i, datafile in enumerate(datafiles):
        path = os.path.join(dir, datafile)
        plt.figure()
        l, n, f, a = np.loadtxt(path, usecols=(0, 1, 2, 3)).T
        print(path)
        delta_nu = np.median(np.diff(f[l == 0]))
        if background is not None:
            dnu = delta_nu
            numax = (dnu / 0.263) ** (1 / 0.772)
            nmax = np.round(numax // dnu) -1
            nx = int(np.round(dnu / fres))
            dnu = nx * fres
            ny = int(np.floor(len(power) / nx))
            print(np.amax(power))
            # nmax = np.argmax(power * (500 <= freq) * (freq <= 1500)) // nx
            numax = (dnu/0.263) ** (1/0.772)
            
            startorder = 0
            endorder = nmax + 5

            start = startorder * nx
            endo = endorder * nx

            apower = power[start:endo]
            pixeldata = np.reshape(apower, (-1, nx))
            plt.imshow(-pixeldata, aspect='auto', cmap='gray',
                        interpolation='gaussian', origin='lower',
                        extent=(0, dnu, start * fres, endo * fres))
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
        plt.plot(np.mod(f[l == 0], delta_nu), f[l == 0], 'bo')
        plt.plot(np.mod(f[l == 1], delta_nu), f[l == 1], 'go')
        plt.plot(np.mod(f[l == 2], delta_nu), f[l == 2], 'yo')
        #plt.plot(np.mod(f[l == 3], delta_nu), f[l == 3], 'mo')
        plt.title(r'The Echelle diagram of %s with $\Delta\nu=$%s' %
                  (starname, delta_nu))
        plt.xlabel(r'Frequency mod $\Delta\nu$ $\mu$Hz]')
        plt.ylabel(r'Frequency [$\mu$Hz]')
        plt.xlim([0, delta_nu])
        plt.savefig('./echelle/%s_echelle_%s_%s.pdf' % (starname, i,
                                                           dnu))
        plt.close()


echelle('HD181096')
