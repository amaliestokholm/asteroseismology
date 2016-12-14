import os
import numpy as np
import scipy
import matplotlib
import seaborn as sns
from collections import namedtuple


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
    matplotlib.rc('text.latex',
                  preamble=r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

matplotlib_setup()

import matplotlib.pyplot as plt


def fix_margins():
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)


ModesBase = namedtuple('Modes', 'l n f inertia error dnu'.split())

class Modes(ModesBase):
    def for_l(self, l):
        mask = self.l == l
        if self.inertia is None:
            inertia = None
        else:
            inertia = self.inertia[mask]
        if self.error is None:
            error = None
        else:
            error = self.error[mask]
        return Modes(self.l[mask], self.n[mask], self.f[mask],
                     inertia, error, self.dnu)

    def for_n(self, n):
        mask = self.n == n
        if self.error is None:
            error = None
        else:
            error = self.error[mask]
        if self.inertia is None:
            inertia = None
        else:
            inertia = self.inertia[mask]
        return Modes(self.l[mask], self.n[mask], self.f[mask],
                     inertia, error, self.dnu)

    def for_ns(self, ns):
        fnl = []
        for n in ns:
            selected = self.for_n(n=n)
            fnl.append(selected.f[0])
        fnl = np.asarray(fnl)
        return fnl
        
    def asarray(self):
        if self.inertia is None:
            inertia = None
        else:
            inertia = np.asarray(self.inertia)
        if self.error is None:
            error = None
        else:
            error = np.asarray(self.error)
        return Modes(l=np.asarray(self.l), n=np.asarray(self.n), f=np.asarray(self.f),
                     inertia=inertia, error=error, dnu=np.asarray(self.dnu))

    def f_as_dict(self):
        keys = zip(self.n, self.l)
        values = self.f
        dictionary = dict(zip(keys, values))
        return dictionary

def kjeldsen_corr(model_modes, observed_modes):
    assert len(observed_modes.n)
    dnu = model_modes.dnu
    dnu_obs = observed_modes.dnu
    # Kjeldsen correction
    # Correcting stellar oscillation frequencies for
    # near-surface effects, Kjeldsen et al., 2008
    bcor = 4.9  # from a solar model
    nu0 = 996
    print('kjeldsen')
    corrected_modes = Modes(l=[], n=[], f=[], inertia=[], error=None, dnu=dnu)
    # inertia_l0 = inertia[l == 0]
    # nl0 = n[l == 0]
    radial_model_modes = model_modes.for_l(l=0)

    plt.figure()
    fix_margins()
    plt.xlabel(r'$\nu_{{model}}$ [$\mu$Hz]')
    plt.ylabel(r'$\nu-\nu_{{model}}$ [$\mu$Hz]')
    color = ['dodgerblue', 'limegreen', 'tomato', 'hotpink']
    ls_obs = [0]  # np.unique(observed_modes.l)
    for l in ls_obs:
        print('l=%s' % l)
        angular_observed_modes = observed_modes.for_l(l=l)
        assert len(angular_observed_modes.n) == len(np.unique(angular_observed_modes.n))
    
        angular_model_modes = model_modes.for_l(l=l)
        inertia_l = angular_model_modes.inertia
        assert len(angular_model_modes.n)
        assert len(angular_observed_modes.n)

        ns = set(angular_model_modes.n) & set(angular_observed_modes.n)
        ns = sorted(ns)
        assert ns
        fnl_ref = angular_model_modes.for_ns(ns)
        fnl_obs = angular_observed_modes.for_ns(ns)
        inertialist = []
        for n in ns:
            selected = angular_model_modes.for_n(n=n)
            inertia_nl, = selected.inertia
            inertia_l0s, = radial_model_modes.inertia[radial_model_modes.n == n]
            inertias = inertia_nl / inertia_l0s
            inertialist.append(inertias)
            corrected_modes.n.append(n)
            corrected_modes.l.append(l)
        inertialist = np.asarray(inertialist)
        r = ((bcor - 1) *
             (bcor * ((fnl_ref) / (fnl_obs)) - ((dnu) / (dnu_obs))) ** (-1))

        #bcor = ((r * ((dnu) / (dnu_obs)) - 1) *
        #       ((r * ((fnl_ref) / (fnl_obs)) - 1) ** (-1)))
        acor = ((np.mean(fnl_obs) - r * np.mean(fnl_ref)) /
               (len(fnl_obs) ** (-1) * np.sum((fnl_obs / nu0) ** bcor)))
        print('a=%s' % acor)
        f_corr = (fnl_ref + (1 / inertialist) * (acor / r) * (fnl_ref / nu0) ** bcor)
        corrected_modes.f.extend(f_corr)
        l = int(l)
        plt.plot(fnl_ref, (fnl_obs - fnl_ref), color=color[l],
                 label=r'l=%s $\nu_{obs}-\nu_{ref}$'% l, marker='d')
        plt.plot(fnl_ref, (f_corr - fnl_ref), color=color[l],
                 label=r'l=%s $\nu_{corr}-\nu_{ref}$'% l, marker='o')
    corrected_modes = corrected_modes.asarray()
    print(corrected_modes)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
               mode="expand", borderaxespad=0., frameon=False)
    plt.savefig('./echelle/kjeldsen_%s.pdf' % (dnu), bbox_inches='tight')
    plt.close()
    print(chisqr(observed_modes, corrected_modes)) 
    return corrected_modes 


def chisqr(observed_modes, corrected_modes):
    observed_dictionary = observed_modes.f_as_dict()
    corrected_dictionary = corrected_modes.f_as_dict()
    nl_keys = sorted(observed_dictionary.keys() & corrected_dictionary.keys())

    f_corr = np.asarray([corrected_dictionary[n, l] for (n,l) in nl_keys])
    f_obs = np.asarray([observed_dictionary[n, l] for (n,l) in nl_keys])

    N = len(f_obs)
    sigma_f_obs = 1  # could be read from mikkelfreq and find median
    return ((1 / N) * np.sum(((f_corr - f_obs) / (sigma_f_obs)) ** 2))


def overplot(job, starfile, obsfile, dnu_obs):
    starname = starfile.replace('.txt', '')
    n_obs, l_obs, f_obs, error_obs = np.loadtxt(
        obsfile, skiprows=1, usecols=(0, 1, 2, 3)).T
    closestfl0_list = []
    dir = './%s/X072669_Y02628_nor/freqs/' % job
    fl0_obs = np.array(sorted(f_obs[l_obs == 0]))
    nl0_obs = np.array(sorted(n_obs[l_obs == 0]))

    datafiles = sorted([s for s in os.listdir(dir) if s.startswith('obs')])
    datafiles = datafiles[7:9]
    observed_modes = Modes(n=n_obs, l=l_obs, f=f_obs,
                           inertia=None, error=error_obs, dnu=dnu_obs)
    for i, datafile in enumerate(datafiles):
        if i % 20 == 0:
            print(i)
        path = os.path.join(dir, datafile)
        l, n, f, inertia = np.loadtxt(path, usecols=(0, 1, 2, 3)).T
        dnu = np.median(np.diff(f[l == 0]))
        model_modes = Modes(l=l, n=n, f=f, inertia=inertia, error=None, dnu=dnu)

        h, plot_position = echelle(starfile, observed_modes.dnu)

        corrected_modes = kjeldsen_corr(model_modes, observed_modes)
        fl0 = np.array(sorted(f[l == 0]))
        nl0 = np.array(sorted(n[l == 0]))
        # closestfl0_index = (min(range(len(fl0)),
        #                     key=lambda x: abs(fl0[x]-fl0_obs[0])))
        # closestfl0_list[i] = fl0[closestfl0_index]
        # closestfl0_list.append(min(fl0, key=lambda p: abs(p - fl0_obs[0])))
        closestfl0_list.append(fl0[nl0 == nl0_obs[0]])
        l0color = 'tomato'  # 'lightcoral'
        l1color = 'firebrick'  # 'crimson'
        plt.plot(*plot_position(closestfl0_list[i]), 'o',
                 color=l0color, markersize=7,
                 label=r'lowest, closest $\nu$ with $l=0$')
        plt.plot(*plot_position(fl0_obs[0]), 'd',
                 color=l0color, markersize=7,
                 label=r'lowest, closest $\nu_{{obs}}$ with $l=0$')
        plt.plot(*plot_position(fl0), 'o', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu$ with $l=0$')
        plt.plot(*plot_position(corrected_modes.f),'*', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu_{corr}$ with $l=0$')
        plt.plot(*plot_position(fl0_obs), 'd', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu_{{obs}}$ with $l=0$')
        # plt.plot(*plot_position(f[l == 1]), 'o', markersize=7,
        #         markeredgewidth=1, markeredgecolor=l1color,
        #         markerfacecolor='none', label=r'$\nu$ with $l=1$')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode="expand", borderaxespad=0., frameon=False)
        plt.savefig('./echelle/%s/%s_echelle_%s_%s.pdf' %
                    (job, starname, i, dnu), bbox_inches='tight')
        plt.close()
        print(closestfl0_list[i], n[closestfl0_list[i] == f], fl0_obs[0])
    minfl0 = min(closestfl0_list, key=lambda p: abs(p - fl0_obs[0]))
    print(closestfl0_list.index(minfl0), minfl0, fl0_obs[0])


def echelle(filename, delta_nu, save=None):
    freq, power = np.loadtxt(filename).T

    fres = (freq[-1] - freq[0]) / (len(freq)-1)
    numax = (delta_nu / 0.263) ** (1 / 0.772)
    nmax = int(np.round(((numax - freq[0]) / delta_nu) - 1))
    nx = int(np.round(delta_nu / fres))
    assert nx % 2 == 0  # we shift by nx/2 pixels below
    dnu = nx * fres

    ny = int(np.floor(len(power) / nx))

    startorder = nmax - 9
    endorder = nmax + 9
    # print("%s pixel rows of %s pixels" % (endorder-startorder, nx))

    start = int(startorder * nx + nx/2)
    endo = int(endorder * nx + nx/2)
    apower = power[start:endo]
    pixeldata = np.reshape(apower, (-1, nx))

    def plot_position(freqs):
        o = freqs - freq[start]
        x = o % dnu
        y = start * fres + dnu * np.floor(o / dnu)
        return x, y

    h = plt.figure()
    fix_margins()
    plt.xlabel(r'Frequency mod $\Delta\nu -\Delta\nu/2$ with $\Delta\nu=$ %s [$\mu$Hz]' % dnu)
    plt.ylabel(r'Frequency [$\mu$Hz]')
    # Subtract half a pixel in order for data points to show up
    # in the middle of the pixel instead of in the lower left corner.
    plt.xlim([-fres/2, dnu-fres/2])
    plt.ylim([start * fres - dnu/2, endo * fres - dnu/2])
    """
    plt.imshow(pixeldata, aspect='auto', cmap='Blues',
               interpolation='gaussian', origin='lower',
               extent=(-fres/2, dnu-fres/2,
                       start * fres - dnu/2, endo * fres - dnu/2))
    """
    if save is not None:
        plt.savefig('./%s_echelle_%s.pdf' % ('181096', delta_nu),
                    bbox_inches='tight')
    return h, plot_position

overplot('amalie2', '181096.txt', 'mikkelfreq.txt', 53.8)
