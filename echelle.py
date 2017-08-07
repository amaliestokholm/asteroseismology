import os
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
from collections import namedtuple


def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    fig_width_pt = 240
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


import matplotlib.pyplot as plt
import seaborn as sns

# Activate Seaborn color aliases
sns.set_palette('colorblind')
sns.set_color_codes(palette='colorblind')
sns.set_context('paper', font_scale=1.7)
sns.set_style("ticks")


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
        return self.attribute_as_dict('f')

    def inertia_as_dict(self):
        return self.attribute_as_dict('inertia')

    def error_as_dict(self):
        return self.attribute_as_dict('error')

    def attribute_as_dict(self, attribute):
        keys = zip(self.n, self.l)
        values = getattr(self, attribute)
        dictionary = dict(zip(keys, values))
        return dictionary

def BG14_corr(model_modes, observed_modes):
    corrected_modes = Modes(l=[], n=[], f=[], inertia=None, error=None, dnu=model_modes.dnu)
    observed_dictionary = observed_modes.f_as_dict()
    model_dictionary = model_modes.f_as_dict()
    error_dict = observed_modes.error_as_dict()
    inertia_dict = model_modes.inertia_as_dict()
    nl_keys = sorted(observed_dictionary.keys() & model_dictionary.keys())
    N = len(nl_keys)

    f_mod = np.asarray([model_dictionary[n, l] for (n,l) in nl_keys])
    f_obs = np.asarray([observed_dictionary[n, l] for (n,l) in nl_keys])
    errors = np.asarray([error_dict[n, l] for (n,l) in nl_keys])
    inertia = 4 * np.pi * np.asarray([inertia_dict[n, l] for (n,l) in nl_keys])
    assert len(f_mod) == len(f_obs) == len(errors) == len(inertia) == N

    matx = np.zeros((N, 2))
    y = (f_obs - f_mod) / errors
    matx[:, 0] = f_mod ** (-1) / (inertia * errors)
    matx[:, 1] = f_mod **  3 / (inertia * errors)

    coeffs = np.linalg.lstsq(matx, y)[0]
    assert coeffs.shape == (2,)
    """
    print(coeffs)
    print(sorted(inertia))
    plt.figure()
    plt.plot(sorted(inertia))
    """
    df = (coeffs[0] * f_mod ** (-1) + coeffs[1] * f_mod ** 3) / inertia
    f_corr = np.asarray(f_mod + df)
    corrected_modes.f.extend(f_corr)
    n, l = zip(*nl_keys)
    corrected_modes.n.extend(n)
    corrected_modes.l.extend(l)
    """
    plt.figure()
    fix_margins()
    plt.xlabel(r'$\nu_{{model}}$ [$\mu$Hz]')
    plt.ylabel(r'$\nu-\nu_{{model}}$ [$\mu$Hz]')
    plt.scatter(f_mod, f_obs - f_mod, c=['rgb'[int(l)] for n, l in nl_keys])
    plt.plot(f_mod, df, 'ko')
    plt.show()
    """
    return corrected_modes, coeffs

def chi(r, a, b, f_mod, f_obs, inertia, errors, nu0):
    f_corr = (f_mod + (1 / inertia) * (a / r) * (f_mod / nu0) ** b)
    return np.mean(((f_corr - f_obs) / (errors)) ** 2)

def chilist(r_list, a_list, *args):
    # chisqr_list = []
    # for r, a in zip(r_list, a_list):
    #     chisqr = chi(r, a, *args)
    #     chisqr_list.append(chisqr)

    # minindex = np.argmin(chisqr_list)
    # return r_list[minindex], a_list[minindex]

    def key(o):
        r, a = o
        return chi(r, a, *args)

    return min(zip(r_list, a_list), key=key)


def chi_optimize(r, a, *args):
    def key(o):
        r, a = o
        # Regularization: force r to be close to 1
        return chi(r, a, *args) + reg(r)

    def reg(r):
        c = 0  # 10**4
        return c * (r-1)**2

    print('Before optimize: chi', chi(r, a, *args), 'Regularization', reg(r))
    res = scipy.optimize.minimize(key, (r,a), options={'disp':True}, method='Nelder-Mead')
    r, a = res.x
    print('After optimize: chi', chi(r, a, *args), 'Regularization', reg(r))
    return r, a


def kjeldsen_corr(model_modes, observed_modes):
    # Kjeldsen correction
    # Correcting stellar oscillation frequencies for
    # near-surface effects, Kjeldsen et al., 2008
    bcor = 4.9  # from a solar model
    nu0 = 996

    assert len(observed_modes.n)
    observed_dictionary = observed_modes.f_as_dict()
    model_dictionary = model_modes.f_as_dict()
    inertia_dict = model_modes.inertia_as_dict()
    error_dict = observed_modes.error_as_dict()
    nl_keys = sorted(observed_dictionary.keys() & model_dictionary.keys())
    N = len(nl_keys)

    dnu = model_modes.dnu
    dnu_obs = observed_modes.dnu
    corrected_modes = Modes(l=[], n=[], f=[], inertia=None, error=None, dnu=dnu)

    f_mod = np.asarray([model_dictionary[n, l] for (n,l) in nl_keys])
    f_obs = np.asarray([observed_dictionary[n, l] for (n,l) in nl_keys])
    errors = np.asarray([error_dict[n, l] for (n,l) in nl_keys])
    # q = np.asarray([inertia_dict[n, 0] for (n,l) in nl_keys])
    # inertia = np.asarray([inertia_dict[n, l] for (n,l) in nl_keys]) / q
    inertia = np.asarray([inertia_dict[n, l] / inertia_dict[n, 0]
                          for n, l in nl_keys])
    assert len(f_mod) == len(f_obs) == N


    r_list = ((bcor - 1) /
          (bcor * ((f_mod) / (f_obs)) - ((dnu) / (dnu_obs))))
    #bcor = ((r * ((dnu) / (dnu_obs)) - 1) *        
    #       ((r * ((f_mod) / (f_obs)) - 1) ** (-1)))
    a_list = ((np.mean(f_obs) - r_list * np.mean(f_mod)) /
            (len(f_obs) ** (-1) * np.sum((f_obs / nu0) ** bcor)))
    rcor = np.mean(r_list)
    acor = np.mean(a_list)
    """
    rcor, acor = chilist(r_list, a_list, bcor, f_mod, f_obs, inertia, errors, nu0)
    print('Before calling minimizer:', rcor, acor)
    rcor, acor = chi_optimize(rcor, acor, bcor, f_mod, f_obs, inertia, errors, nu0)
    print('After calling minimizer:', rcor, acor)
    """
    f_corr = (f_mod + (1 / inertia) * 
              (acor / rcor) * (f_mod / nu0) ** bcor)
    corrected_modes.f.extend(f_corr)
    n, l = zip(*nl_keys)
    corrected_modes.n.extend(n)
    corrected_modes.l.extend(l)
    """
    radial_model_modes = model_modes.for_l(l=0)
    plt.figure()
    fix_margins()
    plt.xlabel(r'$\nu_{{model}}$ [$\mu$Hz]')
    plt.ylabel(r'$\nu-\nu_{{model}}$ [$\mu$Hz]')
    color = ['dodgerblue', 'limegreen', 'tomato', 'hotpink']
    """
    """
    ls_obs = [0]  # np.unique(observed_modes.l)
    for l in ls_obs:
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
        f_corr = (fnl_ref + (1 / inertialist) * (acor / r) * (fnl_ref / nu0) ** bcor)
        corrected_modes.f.extend(f_corr)
        l = int(l)
        plt.plot(fnl_ref, (fnl_obs - fnl_ref), color=color[l],
                 label=r'l=%s $\nu_{obs}-\nu_{ref}$'% l, marker='d')
        plt.plot(fnl_ref, (f_corr - fnl_ref), color=color[l],
                 label=r'l=%s $\nu_{corr}-\nu_{ref}$'% l, marker='o')
    corrected_modes = corrected_modes.asarray()
    """
    """
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
               mode="expand", borderaxespad=0., frameon=False)
    plt.savefig('./echelle/amalie3_kjeldsen/kjeldsen_%s.pdf' % (dnu), bbox_inches='tight')
    plt.close()
    """
    chisqr_value = chisqr(observed_modes, corrected_modes) 
    return corrected_modes, chisqr_value


def chisqr(observed_modes, corrected_modes):
    observed_dictionary = observed_modes.f_as_dict()
    corrected_dictionary = corrected_modes.f_as_dict()
    nl_keys = sorted(observed_dictionary.keys() & corrected_dictionary.keys())

    f_corr = np.asarray([corrected_dictionary[n, l] for (n,l) in nl_keys])
    f_obs = np.asarray([observed_dictionary[n, l] for (n,l) in nl_keys])

    N = len(f_obs)
    error_dict = observed_modes.error_as_dict()
    errors = np.asarray([error_dict[n, l] for (n,l) in nl_keys])
    return ((1 / N) * np.sum(((f_corr - f_obs) / (errors)) ** 2))


def overplot(job, starfile, obsfile, dnu_obs):
    starname = starfile.replace('.txt', '')
    n_obs, l_obs, f_obs, error_obs = np.loadtxt(
        obsfile, skiprows=1, usecols=(0, 1, 2, 3)).T
    closestfl0_list = []
    chisqr_list = []
    dir = './%s/X072669_Y02628_nor/freqs/' % job
    fl0_obs = np.array(sorted(f_obs[l_obs == 0]))
    nl0_obs = np.array(sorted(n_obs[l_obs == 0]))

    datafiles = sorted([s for s in os.listdir(dir) if s.startswith('obs')])
    # datafiles = datafiles[7:9]
    observed_modes = Modes(n=n_obs, l=l_obs, f=f_obs,
                           inertia=None, error=error_obs, dnu=dnu_obs)
    observed_dictionary = observed_modes.f_as_dict()
    for i, datafile in enumerate(datafiles):
        if i % 20 == 0:
            print(i)
        path = os.path.join(dir, datafile)
        l, n, f, inertia = np.loadtxt(path, usecols=(0, 1, 2, 3)).T
        dnu = np.median(np.diff(f[l == 0]))

        model_modes = Modes(l=l, n=n, f=f, inertia=inertia, error=None, dnu=dnu)
        model_dictionary = model_modes.f_as_dict()
        nl_keys = sorted(observed_dictionary.keys() & model_dictionary.keys())
        h, plot_position = echelle(starfile, observed_modes.dnu)

        BG14_corrected_modes, coeffs  = BG14_corr(model_modes, observed_modes)
        HK08_corrected_modes, chisqr = kjeldsen_corr(model_modes, observed_modes)
        chisqr_list.append(chisqr)
        nl0 = np.array(sorted(n[l == 0]))
        HK08_corr_dict = HK08_corrected_modes.f_as_dict()
        BG14_corr_dict = BG14_corrected_modes.f_as_dict()
        f_mod_l0 = np.asarray([model_dictionary[n, l] for (n,l) in nl_keys if l == 0])
        f_obs_l0 = np.asarray([observed_dictionary[n,l] for (n,l) in nl_keys if l == 0])
        f_HK08corr_l0 = np.asarray([HK08_corr_dict[n, l] for (n,l) in nl_keys if l == 0])
        f_BG14corr_l0 = np.asarray([BG14_corr_dict[n, l] for (n,l) in nl_keys if l == 0])
        closestfl0_list.append(f_mod_l0[0])
        print(closestfl0_list)
        l0color = 'tomato'  # 'lightcoral'
        l1color = 'firebrick'  # 'crimson'
        plt.plot(*plot_position(closestfl0_list[i]), 'o',
                 color=l0color, markersize=7,
                 label=r'lowest, closest $\nu$ with $l=0$')
        plt.plot(*plot_position(fl0_obs[0]), 'd',
                 color=l0color, markersize=7,
                 label=r'lowest, closest $\nu_{{obs}}$ with $l=0$')
        plt.plot(*plot_position(f_HK08corr_l0),'*', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu_{HK08 corr}$ with $l=0$')
        plt.plot(*plot_position(f_BG14corr_l0),'s', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu_{BG14 corr}$ with $l=0$')
        plt.plot(*plot_position(f_mod_l0), 'o', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu$ with $l=0$')
        plt.plot(*plot_position(fl0_obs), 'd', markersize=7,
                 markeredgewidth=1, markeredgecolor=l0color,
                 markerfacecolor='none', label=r'$\nu_{{obs}}$ with $l=0$')
        # plt.plot(*plot_position(f[l == 1]), 'o', markersize=7,
        #         markeredgewidth=1, markeredgecolor=l1color,
        #         markerfacecolor='none', label=r'$\nu$ with $l=1$')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode="expand", borderaxespad=0., frameon=False)
        plt.savefig('./echelle/%s/echelle/%s_echelle_%03d_%s.pdf' %
                    (job, starname, i, dnu), bbox_inches='tight')
        plt.close()
        plt.figure()
        # fix_margins()
        plt.xlabel(r'$\nu_{{obs}}$ / $\mu$Hz')
        plt.ylabel(r'$\nu_{obs}-\nu_{{mod}}$ / $\mu$Hz')
        plt.plot(f_obs_l0, (f_obs_l0 - f_mod_l0), color='dodgerblue',
                 label=r'l=%s $\nu_{obs}-\nu_{mod}$'% 0, marker='d', linestyle='None')
        plt.plot(f_obs_l0, (f_obs_l0 - f_HK08corr_l0), color='dodgerblue',
                 label=r'l=%s $\nu_{obs}-\nu_{HK08 corr}$'% 0, marker='*', linestyle='None')
        plt.plot(f_obs_l0, (f_obs_l0 - f_BG14corr_l0), color='dodgerblue',
                 label=r'l=%s $\nu_{obs}-\nu_{BG14 corr}$'% 0, marker='s', linestyle='None')
        plt.plot(f_obs_l0, coeffs[0] * f_mod_l0 ** (-1) + coeffs[1] * f_mod_l0 ** (3))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode="expand", borderaxespad=0., frameon=False)
        plt.savefig('./echelle/%s/correction/%s_correctionplot%03d_%s.pdf' %
                    (job, starname, i, dnu), bbox_inches='tight')
        plt.close()
        print(closestfl0_list[i], n[closestfl0_list[i] == f], fl0_obs[0])
    minfl0 = min(closestfl0_list, key=lambda p: abs(p - fl0_obs[0]))
    minchisqr = min(chisqr_list)
    print(closestfl0_list.index(minfl0), minfl0, fl0_obs[0], chisqr_list.index(minchisqr), minchisqr)


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

    start = int(startorder * nx)
    endo = int(endorder * nx)
    apower = power[start:endo]
    pixeldata = np.reshape(apower, (-1, nx))

    def plot_position(freqs):
        o = freqs - freq[start]
        x = o % dnu
        y = start * fres + dnu * np.floor(o / dnu)
        return x, y

    h = plt.figure()
    plt.xlabel(r'Frequency mod $\Delta\nu$ [$\mu$Hz]' % dnu)
    plt.ylabel(r'Frequency [$\mu$Hz]')
    # Subtract half a pixel in order for data points to show up
    # in the middle of the pixel instead of in the lower left corner.
    plt.xlim([-fres/2, dnu-fres/2])
    plt.ylim([start * fres, endo * fres])
    for row in range(pixeldata.shape[0]):
        bottom = (start + (nx * row)) * fres
        top = (start + (nx * (row + 1))) * fres
        blur_data = ndimage.gaussian_filter(pixeldata[row:row+1], 75)
        plt.imshow(blur_data, aspect='auto', cmap='gray',
                   interpolation='gaussian', origin='lower',
                   extent=(-fres/2, dnu-fres/2, bottom, top))
    if save is not None:
        plt.savefig('./%s_echelle_%s.pdf' % ('181096', delta_nu),
                    bbox_inches='tight')
    return h, plot_position

#overplot('amalie3', '181096.txt', 'mikkelfreq.txt', 53.8)
echelle('HD181096_new.txt', 54, save=1) 
#echelle('HR7322.ts.fft.bgcorr', 54, save=1)
plt.show()
