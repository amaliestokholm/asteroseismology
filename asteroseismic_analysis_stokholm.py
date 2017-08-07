"""
This program is written in Python 3.
In order to have an overview of the code, each part is made as
a local function. This composition is chosen in order to save time.

The user can specify parameters such as the star and the frequency
interval under "INPUTS".
"""
# INPUTS
"""
First, the star needs to be defined.
This is done by defining:
    - 'starname':       The name of the star as a string e.g. 'HD181096'.
    - 'ID':             The ID name for the star as a string e.g.
                        '181096'.
                        This should also be the name of the directory
                        inside './data/ID' in order for the analysis
                        to work.

Then the frequency interval in cyclic frequencies (f=1/P) should be
defined. The first part of the code outputs the Nyquist frequency, and
the upper limit should not exceed this frequency for a correct analysis:
    - 'minfreq':        The lower limit. The noise introduce a very tall
                        peak around 0, so this could preferable be
                        higher than 0.
    - 'maxfreq':        The upper limit. This should not exceed the
                        Nyquist frequency.

Define the constants for the analysis. The constants are:
    - 'quarter':        Chosen period of time.
    - 'kernelsize':     Mediannumber used for the median filter.
                        NB: this must be an odd number.
    - 'sigma':          the limiting sigma for the sigma-clipping.
    - 'gausssigma':     Standard deviation for Gaussian kernel.
    - 'comparorder':    How many point on each side to use to compare in
                        order to find relative maxima (peaks).
    - 'minheight':      Minimum height of saved peaks.
    - 'nmsigma':        Standard deviation for an extremed smoothened
                        power spectrum in order to find nu_max.
    - 'ac_minheight':   Essentially the same as 'minheight', but in the
                        autocorrelated spectrum, the peaks are much
                        lower (between 1 and 0), so this is a smaller
                        value.
    - 'ac_comparorder': Essentially the same as 'comparorder', but in
                        the autocorrelated spectrum, the comparorder
                        should be much higher due to noisy peaks.
    - 'dv':             This is used to find the maximum of the peak in
                        order to find delta_nu by gaussian fitting.
                        It is half the length of the cutted spectrum
                        used in the fitting.
    - 'nu_max_guess':   An initial guess for nu_max used to find the
                        envelope of the oscillations.
"""

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
ac_minheight = 0.07 # 0.15
ac_comparorder = 700
dv = 10
nu_max_guess = 960

frequencies_from = None #and (
    #'/home/amalie/Dropbox/Uddannelse/UNI/1516 - fysik 3. år/' +
    #'Bachelorprojekt/asteroseismology/data/181096/' +
    #'min5_max8490/powerdensity_q1_s4_k801_c1.npz')

# Import the modules
import os
import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.optimize import curve_fit
from time import time as now
import itertools
import matplotlib


def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    # \showthe\columnwidth
    #fig_width_pt = 240
    #inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    #fig_width = fig_width_pt * inches_per_pt
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


import matplotlib.pyplot as plt
import seaborn as sns

# Activate Seaborn color aliases
sns.set_palette('colorblind')
sns.set_color_codes(palette='colorblind')
sns.set_context('paper', font_scale=1.7)
sns.set_style("ticks")


def fix_margins():
    plots.plot_margins()

import kicdata as kic
import ts_powerspectrum as pspec

# Make cryptic abbreviations
minmax = 'min%s_max%s' % (minfreq, maxfreq)
para = 'q%s_s%s_k%s' % (quarter, sigma, kernelsize)
direc = './data/%s/%s' % (ID, minmax)

# Create directory
if not os.path.exists(direc):
    os.makedirs(direc)

# Create filenames and abbreviations of filenames
ts = ('%s/timeseries_%s.npz'
      % (direc, para))
ps = ('%s/power_%s.npz'
      % (direc, para))
pds = ('%s/powerdensity_%s.npz'
       % (direc, para))
cps = ('%s/corrected_power_%s.npz'
       % (direc, para))
sps = ('%s/smoothpower_%s_gs%s.npz'
       % (direc, para, gausssigma))
fp = ('%s/peaks_%s_gs%s_o%s_mh_%s.npz'
      % (direc, para, gausssigma, comparorder, minheight))
dn = ('%s/deltanupeaks_%s_gs%s_o%s_mh_%s.npz'
      % (direc, para, gausssigma, ac_comparorder,
         ac_minheight))


def loadnpz(filename):
    """
    Load compressed data
    """
    return np.load(filename)['data']


def savenpz(filename, data):
    """
    Save compressed data
    """
    return np.savez(filename, data=data)


def npzsavetxt(filename, filename2):
    """
    Save compressed data as txt
    """
    if not isinstance(filename2, str):
        raise TypeError(type(filename2))
    data = loadnpz(filename)
    return np.savetxt(filename2, np.squeeze(data))

print('Chosen star: %s' % starname)


def make_the_timeseries():
    print('Read and filter data')
    # Load data from the star
    time, flux = kic.getdata(ID, kernelsize, quarter, sigma, noisecut)
    #time = time[:((len(time)+1)//2)]
    #flux = flux[:((len(flux)+1)//2)]
    assert len(time) == len(flux)

    # Calculate and print Nyquist-frequency
    dt = np.diff(time)
    nyquist = 1 / (2 * np.median(dt))
    print('Nyquist frequency: %s µHz' % str(nyquist))

    # Plot the time series
    """
    plt.figure()
    plt.plot(time, flux, 'k.')
    plt.xlabel(r'Relative time [Ms]')
    plt.ylabel(r'Photometry')
    plt.xlim([np.amin(time), np.amax(time)])
    plt.savefig('%s_time.pdf' % (starname), bbox_inches='tight')
    """

    # Save data in textfile
    print('Write %d entries to %s' % (len(time), ts))
    timerStart = now()

    savenpz(ts, np.transpose([time, flux]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def make_the_power_spectrum():
    print('Calculate power spectrum')
    # Load data from binary file
    time, flux = loadnpz(ts).T

    # Run the fourier transform (cyclic frequencies in µHz)
    if frequencies_from is not None:
        old_freq = np.load(frequencies_from)['data'][:, 0]
        old_step = np.median(np.diff(old_freq))
        print("Using frequencies from old data. " +
              "Min %g max %g " % (old_freq.min(), old_freq.max()) +
              "step %g" % old_step)
        freq = old_freq[(minfreq <= old_freq) & (old_freq <= maxfreq)]
        freq, power, alpha, beta = pspec.power_spectrum(
            time, flux, freq=freq)
    else:
        oversample = 4  #37.395163336441328  # Step size 0.005
        freq, power, alpha, beta = pspec.power_spectrum(
            time, flux, oversample=oversample,
            memory_use=500000 * 10,
            minfreq=minfreq, maxfreq=maxfreq)

    # Convert powers into ppm^2
    power *= 1e12

    # Plot the power spectrum
    plt.figure()
    # fix_margins
    plt.plot(freq, power, 'k', linewidth=0.1)
    plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.savefig('%s_power_%s_%s.pdf' % (starname, minfreq, maxfreq), bbox_inches='tight')

    plt.figure()
    # fix_margins
    plt.plot(freq, power, 'k', linewidth=0.1)
    plt.xlim([np.amin(freq), 5000])
    plt.title('The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.savefig('%s_zoom_ps_%s_%s.pdf' % (starname, minfreq, maxfreq), bbox_inches='tight')

    # Save data in textfile
    print('Write %d entries to %s' % (len(freq), ps))
    timerStart = now()

    savenpz(ps, np.transpose([freq, power]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def smooth_power_spectrum():
    print('Smoothing of the power spectrum')

    # Load data
    freq, power = loadnpz(ps).T

    # Run Gaussian filter
    smooth_data = scipy.ndimage.gaussian_filter1d(power, sigma=gausssigma)

    # Cut the power spectrum in order to view the oscillations in plots
    """
    freqfilt = freq > 12  # Nok her! Amalie
    freq = freq[freqfilt]
    smooth_data = smooth_data[freqfilt]
    """

    # Plot the smoothened power spectrum
    plt.figure()
    # fix_margins
    plt.plot(freq, smooth_data, 'r-', linewidth=0.1)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.savefig('%s_smoothenpower_%s_%s.pdf' % (starname, minfreq,
                maxfreq), bbox_inches='tight')

    plt.figure()
    # fix_margins
    plt.plot(freq, smooth_data, 'k', linewidth=0.1)
    # plt.title(r'The smoothened power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), 2000])
    plt.savefig('%s_smoothenpower_zoom_%s_%s.pdf' % (starname,
                                                      minfreq, maxfreq), bbox_inches='tight')
    #plt.show()

    # Save data in npz file
    print('Write %d entries to %s' % (len(smooth_data), sps))
    timerStart = now()

    savenpz(sps, np.transpose([freq, smooth_data]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def background(nu_max):
    print('Calculate the power density spectrum')
    # Load power spectrum
    rfreq, rpower = loadnpz(ps).T
    freq, power = loadnpz(sps).T
    time, flux = loadnpz(ts).T
    """ If an oscillation has amplitude A, it will have a peak of A^2 in
    the power spectrum, and A^2 * L in the power density spectrum,
    where L is the length of the time series. """

    # Change the amplitude by multiplying the PS with L in seconds
    L = time[-1] - time[0]
    print('The length of the time series is %s Ms' % L)
    rpowerden = rpower * L
    powerden = power * L

    # Plot
    plt.figure()
    # fix_margins
    plt.plot(freq, powerden, 'k', linewidth=0.5)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\,\mu$Hz$^{-1}$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.savefig('%s_powerden_%s_%s.pdf' % (starname, minfreq, maxfreq), bbox_inches='tight')

    # Save data in textfile
    #print('Write %d entries to %s' % (len(freq), pds))
    #timerStart = now()

    #savenpz(pds, np.transpose([freq, powerden]))

    #elapsedTime = now() - timerStart
    #print('Took %.2f s' % elapsedTime)
    print('Corrects for granulation')
    """ This is based on background modelling due to granulation and
    described in 'Automated extraction of oscillation parameters for
    Kepler observations of solar-type stars' by Huber et al.
    """
    # Load data
    #freq, powerden = loadnpz(pds).T


    # In order to perform the fit, we choose to weight the data by fitting the model to logaritmic bins.
    def running_median(freq, powerden, weights=None, bin_size=None, bins=None):
        if bin_size is not None and bins is not None:
            raise TypeError('cannot specify both bin_size and bins')
        freq = np.squeeze(freq)
        powerden = np.squeeze(powerden)
        n, = freq.shape
        n_, = powerden.shape
        assert n == n_

        if weights is None:
            weights = np.ones(n, dtype=np.float32)

        # Sort data by frequency
        sort_ind = np.argsort(freq)
        freq = freq[sort_ind]
        powerden = powerden[sort_ind]
        weights = weights[sort_ind]

        # Compute log of frequencies
        log_freq = np.log10(freq)
        # Compute bin_size
        if bin_size is None:
            if bins is None:
                bins = 10000
            df = np.diff(log_freq)
            d = np.median(df)
            close = df < 100*d
            span = np.sum(df[close])
            bin_size = span / bins
        bin_index = np.floor((log_freq - log_freq[0]) / bin_size)
        internal_boundary = 1 + (bin_index[1:] != bin_index[:-1]).nonzero()[0]
        boundary = [0] + internal_boundary.tolist() + [n]

        bin_freq = []
        bin_pden = []
        bin_weight = []
        for i, j in zip(boundary[:-1], boundary[1:]):
            bin_freq.append(np.mean(freq[i:j]))
            bin_pden.append(np.median(powerden[i:j]))
            bin_weight.append(np.sum(weights[i:j]))
        return np.array(bin_freq), np.array(bin_pden), np.array(bin_weight)

    # Eq. 1 in mentioned paper
    def background_fit_2(nu, sigma, tau):
        k1 = ((4 * sigma ** 2 * tau) /
              (1 + (2 * np.pi * nu * tau) ** 2 +
               (2 * np.pi * nu * tau) ** 4))
        return k1

    def background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
        k1 = background_fit_2(nu=nu, sigma=sigma_0, tau=tau_0)
        k2 = background_fit_2(nu=nu, sigma=sigma_1, tau=tau_1)
        return P_n + k1 + k2

    def logbackground_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
        assert nu.all() > 0
        assert np.all(np.isfinite(nu)) == True

        xs = background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n)
        invalid = xs <= 0
        xs[invalid] = 1
        log_xs = np.log10(xs)
        log_xs[invalid] = -10000  # return a very low number for log of something negative
        return log_xs
    
    def gridsearch(f, xs, ys, params):
        # Save l2-norm in a dictionary for the tuple of chosen parameters
        score = {}
        dxs = np.diff(np.log10(xs))
        dxs = np.concatenate([dxs, [dxs[-1]]])
        for p in itertools.product(*params):
            print('\rNow %f %f %f %f %f, Done %f' %
                  (*p, len(score)/np.product([len(x) for x in params])),
                  end='')
            zs = f(xs, *p)
            score[p] = np.sum((ys- zs) ** 2)
        print('')
        return min(score.keys(), key=lambda p: score[p])

    P_n = np.arange(0.1, 0.25, step=0.05)  #[np.median(powerden[freq > f]) for f in np.arange(2000, 6000, step=500)]
    guess_sigma_0 =  [n * np.sqrt(np.mean(powerden ** 2)) for n in np.arange(10, 50, step=5)]
    guess_tau_0 = [n * (1 / nu_max) for n in np.arange(0.01, 0.2, step=0.05)]
    guess_sigma_1 = [n * np.sqrt(np.mean(powerden ** 2)) for n in np.arange(10, 50, step=5)]
    guess_tau_1 = [n * (1 / nu_max) for n in np.arange(0.01, 0.2, step=0.05)]

    print('Parameterspace is %f-%f, %f-%f, %f-%f, %f-%f, and %f-%f' % (
        np.min(guess_sigma_0), np.max(guess_sigma_0), np.min(guess_tau_0), np.max(guess_tau_0),
        np.min(guess_sigma_1), np.max(guess_sigma_1), np.min(guess_tau_1), np.max(guess_tau_1),
        np.min(P_n), np.max(P_n)))

    # Cut out around the signals in order not to overfit them
    minimum = 500
    maximum = 1500

    filt = (freq > minimum) & (freq < maximum)
    freq_filt = freq[~filt]
    powerden_filt = powerden[~filt]
    
    freq_filt, powerden_filt, ws = running_median(freq_filt, powerden_filt, bin_size=1e-3)

    def cost(popt):
        return np.mean((logbackground_fit(freq_filt, *popt) - np.log10(powerden_filt)) ** 2)

    freq_fit, powerden_fit, ws = running_median(freq, powerden, bin_size=1e-4)

    z0 = [guess_sigma_0, guess_tau_0, guess_sigma_1, guess_tau_1, P_n]
    popt = gridsearch(logbackground_fit, freq_fit, np.log10(powerden_fit), z0)
    # popt = [52.433858, 0.000885, 81.893752, 0.000167, 0.220056]

    print('Best parameter for background were: s_0 %f t_0 %f s_1 %f t_1 %f P_n %f' % tuple(popt))
    # Fit
    #z0 = [guess_sigma_0, guess_tau_0, guess_sigma_1, guess_tau_1]
    popt, pcov = curve_fit(logbackground_fit, freq_fit,
                           np.log10(powerden_fit), p0=popt, maxfev=10000)
    print('Best parameter for background were: s_0 %f t_0 %f s_1 %f t_1 %f P_n %f' % tuple(popt))

    print('Cost = %f' % cost(popt))
    freq_plot = freq[::1000]
    powerden_plot = powerden[::1000]
    rpowerden_plot = rpowerden[::1000]

    plt.figure()
    plt.loglog(freq_plot, powerden_plot, '0.2', basex=10, basey=10, linewidth=0.5)
    plt.loglog(freq_plot, background_fit(freq_plot, *popt), 'steelblue', linestyle='-', basex=10,
               basey=10)
    plt.loglog(freq_plot, popt[4] + background_fit_2(freq_plot, *popt[:2]), 'steelblue', linestyle='--',
               basex=10, basey=10)
    plt.loglog(freq_plot, popt[4] + background_fit_2(freq_plot, *popt[2:4]), 'steelblue', linestyle='--',
               basex=10, basey=10)
    plt.loglog(freq_plot, np.ones(len(freq_plot)) * popt[4], 'royalblue', linestyle='--')
    # plt.title(r'The power density spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\, \mu$Hz$^{-1}$]')
    plt.xlim([np.amin(freq_plot), np.amax(freq_plot)])
    plt.ylim([10 ** (-2), 2 * 10 ** (2)])
    plt.savefig('%s_backgroundfit_%s_%s.pdf' % (starname,
                minfreq, maxfreq), bbox_inches='tight')

    # Correct for this simulated background by dividing it out
    corr_powerden = rpowerden / background_fit(rfreq, *popt)
    corr_powerden_plot = powerden_plot / background_fit(freq_plot, * popt)

    plt.figure()
    # fix_margins()
    plt.loglog(freq_plot, powerden_plot, '0.75', basex=10, basey=10, linewidth=0.1)
    plt.loglog(freq_plot, corr_powerden_plot, 'k', basex=10, basey=10, linewidth=0.1)
    # plt.title(r'The corrected power density spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\,\mu$Hz$^{-1}$]')
    plt.xlim([np.amin(freq_plot), np.amax(freq_plot)])
    plt.savefig('%s_backgroundcorrected_%s_%s.pdf' % (starname,
                minfreq, maxfreq), bbox_inches='tight')
    #plt.show()

    #time, flux = loadnpz(ts).T

    # Change the amplitude by dividing the PDS with L
    print('The length of the time series is %s Ms' % L)
    corr_power = corr_powerden / L
    corr_power_plot = corr_powerden_plot / L

    plt.figure()
    # fix_margins()
    plt.plot(freq_plot, corr_power_plot, 'k', linewidth=0.1)
    # plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq_plot), 2000])
    plt.savefig('%s_powerspectrum_final_%s_%s.pdf' % (starname,
                minfreq, maxfreq), bbox_inches='tight')
    #plt.show()

    # Save data in textfile
    print('Write %d entries to %s' % (len(corr_power), cps))
    timerStart = now()

    savenpz(cps, np.transpose([freq, corr_power]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def find_peaks(freq, power, minheight, comparorder):
    print('Find peaks with minheight %s and comparorder %s' % (minheight, comparorder))

    # Find relative maxima using scipy.signal.argrelmax
    point = scipy.signal.argrelmax(power, order=comparorder)
    if len(point[0]) == 0:
        raise Exception("No maxima found")
    print(len(point[0]))
    # Find location and height of peaks
    peak = freq[point]
    height = power[point]

    # Only peaks over a given height 'minheight' should be included
    included_peak = peak[height > minheight]
    included_height = height[height > minheight]
    print(len(peak), len(included_peak))

    if len(included_peak) == 0:
        raise Exception(
            "No peaks are above minheight %s, " % minheight +
            "highest is %s" % height.max())

    return included_peak, included_height


def gauss(x, A, mean, sigma, y0):
    return y0 + A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def gaussian_fit(xdata, ydata):
    a = np.amax(ydata)
    mean = xdata[np.argmax(ydata)]
    sigma = np.std(xdata)
    y0 = np.amin(ydata)
    popt, pcov = curve_fit(gauss, xdata, ydata, p0=[a, mean, sigma, y0])
    return popt


def find_numax():
    # Not used
    """
    This finds nu_max using extreme smoothing. nu_max is the
    position of the top of gaussian envelope of the power spectrum, so
    by smoothing the shape of the envelope becomes clear and nu_max
    is easy to find.
    """
    print('Find nu_max')

    # Load data
    freq, power = loadnpz(cps).T

    ac_minheights = ac_minheight

    # Run Gaussian filter
    nmps = scipy.ndimage.gaussian_filter1d(power, 1.5*nmsigma)

    # Initial guess of nu_max
    included_peak, included_height = find_peaks(freq, nmps,
                                                ac_minheights,
                                                ac_comparorder)
    max_peak = np.argmax(included_height)
    print('nu_max is guessed to be %s' % included_peak[max_peak])
    nu_max_filt = (50 <= freq) & (2000 >= freq)
    freqcut = freq[nu_max_filt]
    nmpscut = nmps[nu_max_filt]
    popt = gaussian_fit(freqcut, nmpscut)
    nu_max = popt[1]
    print('nu_max = %.4f, popt = %s' % (nu_max, popt))

    A = np.amax(nmps[nu_max_filt]) / np.amax(power[nu_max_filt])

    # Plot the smoothened power spectrum and the value of nu_max
    plt.figure()
    # fix_margins
    plt.plot(freq[::100], A * power[::100], 'k', linewidth=0.2)
    plt.plot(freq[::100], nmps[::100], 'k')
    plt.plot(freqcut, nmpscut, 'g')
    plt.plot(freqcut, gauss(freqcut, * popt), 'b')
    plt.plot(nu_max, gauss(nu_max, * popt), 'ro')
    # plt.title(r'The extremely smoothened power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), 2000])
    plt.ylim([0, 0.75])
    plt.savefig('%s_nm%s_%s_%s.pdf' % (starname, nmsigma, minfreq,
                maxfreq), bbox_inches='tight')
    #plt.show()

    return nu_max


def acf(x, nlags, fft=False, norm=True):
    '''
    This is a modified module of statsmodels.tsa.stattools.acf from
    http://statsmodels.sourceforge.net/stable/index.html,
    which is statistical module specific for time series.

    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    nlags: int, optional
        Number of lags to return autocorrelation for.

    Returns
    -------
    acf : array
        autocorrelation function

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.
    '''
    assert fft
    from statsmodels.compat.scipy import _next_regular

    nobs = len(x)
    d = nobs  # changes if unbiased
    x = np.squeeze(np.asarray(x))
    #JP: move to acovf
    x0 = x - x.mean()
    # ensure that we always use a power of 2 or 3 for zero-padding,
    # this way we'll ensure O(n log n) runtime of the fft.
    n = _next_regular(2 * nobs + 1)
    Frf = np.fft.fft(x0, n=n)  # zero-pad for separability
    acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d
    if norm:
        acf /= acf[0]
    return np.real(acf[:nlags + 1])


def find_deltanu_and_nu_max():
    print('Find delta_nu and nu_max')

    # Load data
    freq, spower = loadnpz(cps).T

    # Determine the number of lags
    lags = np.ceil(len(freq)/2)
    if not os.path.exists(dn):
        print('Run autocorrelation')
        # Run autocorrelation, the argument has to be a np-array.
        corr = acf(spower, nlags=lags, fft=True)

        # Write data to textfile
        print('Write %d entries to %s' % (len(corr), dn))
        timerStart = now()

        savenpz(dn, np.transpose([corr]))

        elapsedTime = now() - timerStart
        print('Took %.2f s' % elapsedTime)

    autocorr = loadnpz(dn).reshape(-1, 1)
    # Make an evenly spaced list in order to plot the peaks
    step = np.mean(np.diff(freq))
    nautocorr = (np.arange(len(autocorr)) * step).reshape(-1, 1)

    # Find peaks in the autocorrelated spectrum
    included_peak, included_height = find_peaks(nautocorr.ravel(),
                                                autocorr.ravel(),
                                                ac_minheight,
                                                ac_comparorder)
    print(nautocorr.shape, autocorr.shape, included_peak.shape)
    print('Peaks found: %s' % len(included_peak))

    # Find delta_nu using a gaussian fit
    guess_delta_nu = included_peak[1]
    print('Delta_nu is guessed to be %s' % guess_delta_nu)
    delta_nu_filt = ((guess_delta_nu - dv <= nautocorr) &
                     (guess_delta_nu + dv >= nautocorr))
    freqcut = nautocorr[delta_nu_filt]
    heightcut = autocorr[delta_nu_filt]

    popt = gaussian_fit(freqcut, heightcut)
    delta_nu = popt[1]
    pm_dn = popt[2]
    print('Fitted delta_nu = %.2f \N{PLUS-MINUS SIGN} %.2f'
          % (delta_nu, pm_dn))

    # Plot autocorrelated spectrum
    plt.figure()
    # fix_margins
    plt.plot(nautocorr, autocorr, 'k', linewidth=0.2)
    plt.plot(included_peak, included_height, 'b.')
    plt.plot(freqcut, heightcut, 'g', linewidth=0.1)
    plt.plot(freqcut, gauss(freqcut, *popt), 'b', linewidth=0.1)
    plt.plot(delta_nu, gauss(delta_nu, *popt), 'r.')
    plt.xlim([np.amin(nautocorr), 1000])
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Autocorrelated function')
    plt.ylim([0, 0.5])
    plt.savefig('dn%s_%s_%s.pdf' % (starname, minfreq, maxfreq), bbox_inches='tight')

    # Find nu_max using the method found in
    # http://arxiv.org/pdf/0910.2764v1.pdf
    cutlimit = 3 * delta_nu
    overlap = delta_nu
    startlist = np.arange(freq[0], freq[-1] - cutlimit, cutlimit - overlap)
    # Discard last subsection for which we don't have enough.
    #startlist = startlist[:-1]
    freqlist = []
    spacinglist = []
    clist = []
    df = freq[1] - freq[0]
    assert np.allclose(np.diff(freq), df)
    for start in startlist:
        stop = start + cutlimit
        #  start = i * df + freq[0]. Solve for i:
        i = int((start - freq[0]) / df)
        # stop = j * df + freq[0]
        j = int((stop - freq[0]) / df)
        chunk = spower[i:j]
        assert len(chunk) == j - i
        freqchunk = freq[i:j]
        # chunk = chunk - np.mean(chunk)  # Already done in acf
        corr = acf(chunk, nlags=len(chunk)-1, fft=True, norm=False)
        assert len(corr) == len(chunk)
        assert len(corr) == j - i
        cf = (stop + start) / 2
        freqlist.append((np.zeros(j - i) + cf))
        spacinglist.append((np.arange(j - i) * df))
        clist.append(corr)

    minlen = min(len(c) for c in clist)
    freqlist = np.asarray([l[:minlen] for l in freqlist])
    spacinglist = np.asarray([s[:minlen] for s in spacinglist])
    clist = np.asarray([c[:minlen] for c in clist])
    fig = plt.figure()
    # fix_margins
    plt.xlabel(r'Central frequency [$\mu$Hz]')
    plt.ylabel(r'Spacing [$\mu$Hz]')
    plt.xlim([freqlist[0, 0], freqlist[-1, 0]])
    plt.imshow(-clist.T, cmap='gray', interpolation='bilinear',
               origin='lower', aspect='auto',
               extent=[freqlist[0, 0], freqlist[-1, 0], 0, spacinglist[0, -1]])
    plt.savefig(r'%s_numax_huber_2_%s_%s.pdf' % (starname,
                cutlimit, overlap), bbox_inches='tight')

    # Collapse the ACF
    collacf = np.sum(clist, axis=1)
    xs = np.linspace(freqlist[0, 0], freqlist[-1, 0], 10000)
   
    #freqs = freqlist[:, 0]
    #print(freqs)
    #filt = (freqs > 500) & (freqs < 1500)
    #print(freqs[filt])
    popt = gaussian_fit(freqlist[:, 0], collacf)

    nu_max = popt[1]
    print(nu_max) 
    nu_max = freqlist[np.argmax(collacf), 0]
    print(nu_max)
    fig = plt.figure()
    # fix_margins
    plt.xlabel(r'Central frequency [$\mu$Hz]')
    plt.ylabel(r'Collapsed ACF [Arbitary Units]')
    plt.xlim([freqlist[0, 0], freqlist[-1, 0]])
    xs = np.linspace(freqlist[0, 0], freqlist[-1, 0], 1000)
    plt.plot(freqlist[:, 0], collacf, 'ko--')
    #plt.plot(xs, gauss(xs, *popt), 'b')
    #loc, scale = skewnorm.fit(collacf)
    #pdf = skewnorm.pdf(xs, 1, loc=loc, scale=scale)
    #plt.plot(xs, pdf, 'k')
    plt.savefig(r'%s_numax_huber_3_%s_%s.pdf' % (starname,
                cutlimit, overlap), bbox_inches='tight')
    plt.show()

    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlabel(r'Correlation')
    Pmesh, phimesh = np.meshgrid(cf, spacing)
    ax.plot_surface(Pmesh, phimesh, a, rstride=10, cstride=10,
                    color='c', linewidth=0.01, antialiased=False)
    plt.show()
    """
    return delta_nu, nu_max


def scalingrelation():
    print('Calculates the radius ')
    # Solar values from the literature
    nu_max_sun = 3090
    delta_nu_sun = 135.1
    T_eff_sun = 5777
    s_nu_max_sun = 30
    s_delta_nu_sun = 0.1
    # From the litterature
    #T_eff = 6347
    #s_teff = 46
    # From the interferometric analysis
    #T_eff = 6211
    #s_teff = 91
    T_eff = 6242
    s_teff = 130
    # From half the timeseries
    #s_nmax = 21
    #s_dn = 0.4 
    # From Dan
    s_nmax = 15.1
    s_dn = 0.2

    # Find asteroseismic parameters
    delta_nu, nu_max = find_deltanu_and_nu_max()
    #delta_nu = 53.92
    #nu_max = 960.2

    # Calculate the radius
    r_nu_max = (nu_max / nu_max_sun)
    r_delta_nu = (delta_nu / delta_nu_sun)
    r_T_eff = (T_eff / T_eff_sun)
    r_R = r_nu_max * r_delta_nu ** (-2) * r_T_eff ** (0.5)

    # Calculate the uncertainty
    dRdnmax = ((1 / nu_max_sun) * (delta_nu / delta_nu_sun) ** (-2) *
               (T_eff / T_eff_sun) ** (0.5))
    dRddn = ((nu_max / nu_max_sun) *
             ((-2 * delta_nu_sun ** 2) / delta_nu ** 3)
             * (T_eff / T_eff_sun) ** (0.5))
    dRdteff = ((nu_max / nu_max_sun) *
               (delta_nu / delta_nu_sun) ** (-2) *
               (T_eff / T_eff_sun) ** (0.5) / (2 * T_eff))
    dRdnmaxsun = (-(nu_max / (nu_max_sun ** 2)) *
                 (delta_nu / delta_nu_sun) ** (-2) *
                 (T_eff / T_eff_sun) ** (0.5))
    dRddnsun = ((nu_max / nu_max_sun) *
                ((2 * delta_nu_sun) / (delta_nu ** 2)) *
                (T_eff / T_eff_sun) ** (0.5))
    s_R = np.sqrt(s_nmax ** 2 * dRdnmax ** 2 + s_dn ** 2 * dRddn ** 2 +
                  s_teff ** 2 * dRdteff ** 2 +
                  s_nu_max_sun ** 2 * dRdnmaxsun ** 2 +
                  s_delta_nu_sun ** 2 * dRddnsun ** 2
                  )
    print(r_R)
    print(s_R)


# Make a zoomed-in plot
def plot_ps():
    freq, spower = loadnpz(cps).T
    dan_freq, dan_power = np.loadtxt('HR7322.ts.fft.bgcorr').T

    minimum = 500
    maximum = 1500

    delta_nu = 53.8
    #filt = (freq > minimum) & (freq < maximum)
    #freq = freq[filt]
    #spower = spower[filt]

    #peak, height = echelle(delta_nu, freq, spower)

    plt.figure()
    # fix_margins()
    color = 'dodgerblue'
    plt.plot(dan_freq, dan_power, c='k', linewidth=1)
    plt.plot(freq, spower, c=color, linewidth=1)
    #plt.plot(peak, height, 'ro')

    #n, l, f = np.loadtxt('10005473fre.txt', skiprows=1, usecols=(0, 1, 2, )).T
    #timpeak = np.loadtxt('181096.pkb', usecols=(2,)).T
    peak = np.loadtxt('mikkelfreq.txt', usecols=(2,)).T
    # plt.plot(f, np.ones(len(f)) * 2, 'bo')
    plt.plot(peak, np.ones(len(peak)) * 2, 'r.')
    
    # for freq in f:
    #    plt.axvline(x=freq, color='b', linestyle='-')
    #for peak in amaliepeak:
    #    plt.axvline(x=peak, color='r', linestyle='-')
    
    plt.xlim([minimum, maximum])
    # plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\,\mu$Hz$^{-1}$]')
    plt.savefig('zoom_cps%s_%s_%s_min_%smax_%s.png' %
                (starname, minfreq, maxfreq, minimum, maximum), dpi=300, bbox_inches='tight')
    plt.show()
    #print(np.transpose([peak, np.round((peak/delta_nu)-1)]))


# Make an Échelle diagram (not used anymore, use a different routine)
def echelle(delta_nu, freq, power):
    print('Plot Échelle diagram')

    e_comparorder = 700
    e_minheight = 0.17 * np.amax(power)

    peak, height = find_peaks(freq, power, e_minheight, e_comparorder)
    """
    peakmod = np.mod(peak, delta_nu)

    n, l, f = np.loadtxt('10005473fre.txt', skiprows=1, usecols=(0, 1, 2, )).T
    #timpeak = np.loadtxt('181096.pkb', usecols=(2,)).T
    amaliepeak = np.loadtxt('181096.txt', usecols=(2,)).T

    plt.figure()
    fix_margins
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
    """
    return peak, height


if __name__ == "__main__":
    matplotlib_setup()
    # Here the functions are called
    make_the_timeseries()
    #make_the_power_spectrum()
    # smooth_power_spectrum()
    # background(nu_max_guess)
    # scalingrelation()
    # plot_ps()
    #npzsavetxt(ts, ('%s/timeseries_%s.txt' % (direc, para)))
    #npzsavetxt(ps, ('%s/power_%s.txt' % (direc, para)))
    #npzsavetxt(ts, ('HR7322_timeseries.txt'))
