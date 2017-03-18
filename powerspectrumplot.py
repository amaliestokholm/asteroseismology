
# Import the modules
import os
import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.optimize import curve_fit
from time import time as now
import matplotlib
import seaborn as sns

def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    fig_width_pt = 328 * 2.1
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('figure', figsize=fig_size)
    matplotlib.rc('font', size=22, family='serif')
    matplotlib.rc('axes', labelsize=22)
    matplotlib.rc('legend', fontsize=22)
    matplotlib.rc('xtick', labelsize=22)
    matplotlib.rc('ytick', labelsize=22)
    matplotlib.rc('text.latex',
                  preamble=r'\usepackage[T1]{fontenc}\usepackage{lmodern}')


matplotlib_setup()

import matplotlib.pyplot as plt


def fix_margins():
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)

# Choose the star:
ID = '181096'
starname = 'HD' + ID


# Define the frequency interval in cyclic frequencies (f=1/P).
minfreq = 5
maxfreq = 8490  # Nyquist for HD181096: 8496 µHz

# Constants for the analysis
quarter = 1
kernelsize = 801
sigma = 4
noisecut = -0.0002
gausssigma = 70  # FWHM~3*delta_nu, gausssigma = FWHM / 2*sqrt(2*ln(2))
comparorder = 500
minheight = 0.75
nmsigma = 7500
ac_minheight = 0.15
ac_comparorder = 700
dv = 10
nu_max_guess = 1000
minmax = 'min%s_max%s' % (minfreq, maxfreq)
para = 'q%s_s%s_k%s' % (quarter, sigma, kernelsize)
direc = './data/%s/%s' % (ID, minmax)
cps = ('%s/corrected_power_%s.npz'
    % (direc, para))


def loadnpz(filename):
    """
    Load compressed data
    """
    return np.load(filename)['data']


# Make a zoomed-in plot
def plot_ps():
    freq, spower = loadnpz(cps).T

    minimum = 500
    maximum = 1500

    delta_nu = 53.8
    filt = (freq > minimum) & (freq < maximum)
    freq = freq[filt]
    spower = spower[filt]

    #peak, height = echelle(delta_nu, freq, spower)

    plt.figure()
    fix_margins()
    color = 'dodgerblue'
    plt.plot(freq, spower, c=color, linewidth=1)
    #plt.plot(peak, height, 'ro')

    #n, l, f = np.loadtxt('10005473fre.txt', skiprows=1, usecols=(0, 1, 2, )).T
    #timpeak = np.loadtxt('181096.pkb', usecols=(2,)).T
    amaliepeak = np.loadtxt('mikkelfreq.txt', usecols=(2,)).T
    # plt.plot(f, np.ones(len(f)) * 2, 'bo')
    plt.plot(amaliepeak, np.ones(len(amaliepeak)) * 0.1, 'ro')
    
    # for freq in f:
    #    plt.axvline(x=freq, color='b', linestyle='-')
    #for peak in amaliepeak:
    #    plt.axvline(x=peak, color='r', linestyle='-')
    
    plt.xlim([minimum, maximum])
    # plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.savefig('zoom_cps%s_%s_%s_min_%smax_%s.png' %
                (starname, minfreq, maxfreq, minimum, maximum))
    plt.show()
    #print(np.transpose([peak, np.round((peak/delta_nu)-1)]))
plot_ps()
