import os
import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple


def matplotlib_setup():
    # \showthe\columnwidth
    # fig_width_pt = 240  #onecolumn mnras:504
    # inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    # fig_width = fig_width_pt * inches_per_pt
    fig_width = 9.96
    posterfont = 30
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('figure', figsize=fig_size)
    matplotlib.rc('font', size=posterfont, family='serif')
    matplotlib.rc('axes', labelsize=posterfont)
    matplotlib.rc('legend', fontsize=posterfont)
    matplotlib.rc('xtick', labelsize=posterfont)
    matplotlib.rc('ytick', labelsize=posterfont)
    matplotlib.rc('text.latex',
                  preamble=r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

# matplotlib_setup()

# import matplotlib.pyplot as plt
# import plots

# Activate Seaborn color aliases
sns.set_palette('colorblind')
sns.set_color_codes(palette='colorblind')
sns.set_context('poster',font_scale=1.7)
sns.set_style("ticks")


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
    'nu_vis.txt', skiprows=1, usecols=(0, 1, 2, 3, 4)).T

ldcoeff = 0.52
angdia = 0.458

plt.figure()
plt.xlabel(r'Spatial frequency [rad$^{-1}$]')
plt.ylabel(r'Visibility$^2$')
scale = 3
plt.errorbar(spatialfreq, v2, yerr=sigma_v2,
             ecolor='b',
             alpha=0.5,
             elinewidth=1 * scale, 
             capsize=2 * scale,
             linestyle='None')
plt.plot(spatialfreq, v2, '.', color='k', alpha=0.7, markersize=4 * scale,
        #mew=0.5 * scale, mfc='None'
        )
spatialfreq = np.asarray(sorted(spatialfreq))
xs = np.linspace(0, 6 * 10 ** 8, 150)
plt.xlim([np.min(xs), np.max(xs)])
ld = limb_darkened_fit(xs, angdia, ldcoeff)
plt.plot(xs, ld, 'k', linewidth=2)
plt.savefig('visibilityplot.pdf', dpi=300, bbox_inches='tight')
#plt.show()
