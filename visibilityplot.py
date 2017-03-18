import os
import numpy as np
import scipy
import matplotlib
import seaborn as sns
from collections import namedtuple


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


def mas2rad(angdia):
    return angdia * 10 ** (-3) * (1/(3600)) * np.pi/180


def limb_darkened_fit(spatialfreq, angdia, ldcoeff):
    a = ((1-ldcoeff)/(2) + (ldcoeff)/(3)) ** (-1)
    x = np.pi * spatialfreq * mas2rad(angdia)
    b = ((1 - ldcoeff) * ((scipy.special.jv(1,x)) / x) +
          ldcoeff * ((np.pi/2) ** (1/2)) *
          ((scipy.special.jv((3/2), x)) / (x ** (3/2))))
    return (a * b) ** 2

spatialfreq, v2, sigma_v2, u, v = np.loadtxt(
    'mean2', skiprows=1, usecols=(0, 1, 2, 3, 4)).T

ldcoeff = 0.52
angdia = 0.458

plt.figure()
fix_margins()
plt.xlabel(r'Spatial frequency [rad$^{-1}$]')
plt.ylabel(r'Visibility$^2$')
color = 'dodgerblue'
plt.errorbar(spatialfreq, v2, yerr=sigma_v2,
             color=color, marker='.', linestyle='None', capthick=1)
spatialfreq = np.asarray(sorted(spatialfreq))
plt.xlim([spatialfreq[0], spatialfreq[-1]])
ld = limb_darkened_fit(spatialfreq, angdia, ldcoeff)
plt.plot(spatialfreq, ld, 'r')
plt.savefig('visibilityplot.png')
plt.show()
