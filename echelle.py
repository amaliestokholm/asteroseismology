import os
import numpy as np
import scipy
import matplotlib
import seaborn as sns


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


def kjeldsen_corr(l, n, f, q, dnu, nu0, obs_n, obs_l, obs_f, delta_nu):
    # Kjeldsen correction
    b = 4.90

    output = []
    llist = []
    i_l0 = q[l == 0]
    nl0 = n[l == 0]
    plt.figure()
    # plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)

    sns.set_style("darkgrid")
    plt.xlabel(r'$\nu_{\text{model}}$ [$\mu$Hz]')
    plt.ylabel(r'$\nu-\nu_{\text{model}}$ [$\mu$Hz]')
    color = ['b', 'g', 'y', 'm']
    uniquelist = np.unique(obs_l)
    for k in uniquelist[:-1]:
        print('l=%s' % k)
        obs_fl = obs_f[obs_l == k]
        obs_nl = obs_n[obs_l == k]
        assert len(obs_nl) == len(np.unique(obs_nl))
        fl = f[l == k]
        nl = n[l == k]
        qsl = q[l == k]
        fobs = []
        fbest = []
        qlist = []

        ns = set(nl) & set(obs_nl)
        for m in sorted(ns):
            aelement, = obs_fl[obs_nl == m]
            element, = fl[nl == m]
            fobs.append(aelement)
            fbest.append(element)
            qs, = qsl[nl == m]
            i_l0s, = i_l0[nl0 == m]
            qs = qs / i_l0s
            qlist.append(qs)
        fbest = np.asarray(fbest)
        fobs = np.asarray(fobs)
        qlist = np.asarray(qlist)
        r = ((b - 1) *
             (b * ((fbest) / (fobs)) - ((dnu) / (delta_nu))) ** (-1))
        a = ((np.mean(fobs) - r * np.mean(fbest)) /
             (len(fobs) ** (-1) * np.sum((fobs / nu0) ** b)))
        fcorr = (fbest + a * (fbest / nu0) ** b) / qlist
        output.append(fcorr)
        k = int(k)
        llist.append(k)
        print(k, color[k])
        plt.plot(fbest, (fobs - fbest), color=color[k], marker='d')
        plt.plot(fbest, (fcorr - fbest), color=color[k], marker='o')

    plt.savefig('./echelle/kjeldsen_%s.pdf' % (dnu))
    plt.close()
    return output, llist


def chisqr(obs_freq, corr_freq):
    N = len(obs_freq)
    sigma_obs_freq = 5
    return ((1 / N) * np.sum(((obs_freq - corr_freq) /
            (sigma_obs_freq)) ** 2))


def overplot(starfile):
    starname = starfile.replace('.txt', '')
    n_obs, l_obs, f_obs = np.loadtxt('amaliefreq.txt').T
    dnu_obs = 53.8
    fl0_obs = sorted(f_obs[l_obs == 0])
    closestfl0_list = []
    dir = './models/X072669_Y02628_nor/freqs/'

    color = ['b', 'g', 'y', 'm']

    datafiles = sorted([s for s in os.listdir(dir) if s.startswith('obs')])
    datafiles = datafiles[800:]
    for i, datafile in enumerate(datafiles):
        if i % 20 == 0:
            print(i)
        path = os.path.join(dir, datafile)
        l, n, f, q = np.loadtxt(path, usecols=(0, 1, 2, 3)).T

        dnu = np.median(np.diff(f[l == 0]))
        h, nu0 = echelle(starfile, dnu)

        # fcorr, llist = kjeldsen_corr(l, n, f, q, dnu, nu0, n_obs,
        #   l_obs, f_obs, dnu_obs)

        fl0 = sorted(f[l == 0])
        # closestfl0_index = (min(range(len(fl0)),
        #                     key=lambda x: abs(fl0[x]-fl0_obs[0])))
        # closestfl0_list[i] = fl0[closestfl0_index]
        closestfl0_list.append(min(fl0, key=lambda p: abs(p - fl0_obs[0])))

        plt.plot(np.mod(fl0_obs[0], dnu), fl0_obs[0], 'ro')
        plt.plot(np.mod(closestfl0_list[i], dnu), closestfl0_list[i], 'md')
        plt.plot(np.mod(fl0, dnu), fl0, 'bo')
        plt.plot(np.mod(fl0_obs, dnu), fl0_obs, 'b', marker='d')
        # for abe, kat in zip(fcorr, llist):
        #    plt.plot(np.mod(abe, dnu), abe, color=color[kat], marker='d')
        plt.plot(np.mod(f[l == 1], dnu), f[l == 1], 'go')
        # plt.plot(np.mod(f[l == 2], dnu), f[l == 2], 'yo')
        # plt.plot(np.mod(f[l == 3], dnu), f[l == 3], 'mo')
        plt.savefig('./echelle/%s_echelle_%s_%s.pdf' % (starname, i, dnu))
        plt.close()
    minfl0 = min(closestfl0_list)
    print(closestfl0_list.index(minfl0), minfl0, fl0_obs[0])


def echelle(filename, delta_nu, save=None):
    freq, power = np.loadtxt(filename).T

    nu0 = freq[np.argmax(power)]
    fres = (freq[-1] - freq[0]) / (len(freq)-1)
    numax = (delta_nu / 0.263) ** (1 / 0.772)
    nmax = int(np.round(((numax - freq[0]) / delta_nu) - 1))
    nx = int(np.round(delta_nu / fres))
    dnu = nx * fres

    ny = int(np.floor(len(power) / nx))

    startorder = nmax - 9
    endorder = nmax + 9

    start = int(startorder * nx)
    endo = int(endorder * nx)
    apower = power[start:endo]
    pixeldata = np.reshape(apower, (-1, nx))

    h = plt.figure()
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    plt.xlabel(r'Frequency mod $\Delta\nu=$ %s [$\mu$Hz]' % dnu)
    plt.ylabel(r'Frequency [$\mu$Hz]')
    plt.xlim([0, dnu])
    plt.ylim([start * fres, endo * fres])
    plt.imshow(-pixeldata, aspect='auto', cmap='gray',
               interpolation='gaussian', origin='lower',
               extent=(0, dnu, start * fres, endo * fres))
    if save is not None:
        plt.savefig('./%s_echelle_%s.pdf' % ('181096', delta_nu))
    return h, nu0

overplot('181096.txt')
