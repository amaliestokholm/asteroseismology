""" This program is written in Python 3.
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
starname = 'HD181096'
ID = '181096'

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
nu_max_guess = 995

# Import the modules
import os
import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.optimize import curve_fit
from time import time as now
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

from mpl_toolkits.mplot3d import Axes3D

import plots
# from statsmodels.tsa.stattools import acf

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
    data = loadnpz(filename)
    return np.savetxt(filename2, data)

print('Chosen star: %s' % starname)


def make_the_timeseries():
    print('Read and filter data')
    # Load data from the star
    time, flux = kic.getdata(ID, kernelsize, quarter, sigma, noisecut)

    # Calculate and print Nyquist-frequency
    dt = np.diff(time)
    nyquist = 1 / (2 * np.median(np.diff(time)))
    print('Nyquist frequency: %s µHz' % str(nyquist))

    # Plot the time series
    plt.figure()
    plt.plot(time, flux, 'k.')
    plt.xlabel(r'Relative time [Ms]')
    plt.ylabel(r'Photometry')
    plt.xlim([np.amin(time), np.amax(time)])
    plots.plot_margins()
    plt.savefig('%s_time.pdf' % (starname))

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
    freq, power, alpha, beta = pspec.power_spectrum(time, flux,
                                                    minfreq=minfreq,
                                                    maxfreq=maxfreq)

    # Convert powers into ppm^2
    power *= 1e12

    # Plot the power spectrum
    plt.figure()
    plt.plot(freq, power, 'k')
    plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.savefig('%s_power_%s_%s.pdf' % (starname, minfreq, maxfreq))

    plt.figure()
    plt.plot(freq, power, 'k', linewidth=0.25)
    plt.xlim([np.amin(freq), 5000])
    plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.savefig(r'%s_zoom_ps_%s_%s.pdf' % (starname, minfreq, maxfreq))

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
    smooth_data = scipy.ndimage.gaussian_filter1d(power, gausssigma)

    # Cut the power spectrum in order to view the oscillations in plots
    freqfilt = freq > 12
    freq = freq[freqfilt]
    smooth_data = smooth_data[freqfilt]

    # Plot the smoothened power spectrum
    plt.figure()
    plt.plot(freq, smooth_data, 'r-', linewidth=0.1)
    # plt.title(r'The smoothened power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.ylim([0, 10])
    plots.plot_margins()
    plt.savefig(r'%s_smoothenpower_%s_%s.pdf' % (starname, minfreq,
                maxfreq))

    plt.figure()
    plt.plot(freq, smooth_data, 'k', linewidth=0.25)
    # plt.title(r'The smoothened power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), 2000])
    plt.ylim([0, 10])
    plots.plot_margins()
    plt.savefig(r'%s_smoothenpower_zoom_%s_%s.pdf' % (starname,
                                                      minfreq, maxfreq))
    #plt.show()

    # Save data in npz file
    print('Write %d entries to %s' % (len(smooth_data), sps))
    timerStart = now()

    savenpz(sps, np.transpose([freq, smooth_data]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def power_density_spectrum():
    print('Calculate the power density spectrum')
    # Load power spectrum
    freq, power = loadnpz(sps).T
    time, flux = loadnpz(ts).T
    """ If an oscillation has amplitude A, it will have a peak of A^2 in
    the power spectrum, and A^2 * L in the power density spectrum,
    where L is the length of the time series. """

    # Change the amplitude by multiplying the PS with L in seconds
    L = time[-1]
    print('The length of the time series is %s Ms' % L)
    powerden = power * L

    # Plot
    plt.figure()
    plt.plot(freq, powerden, 'k', linewidth=0.1)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\, \mu$Hz^{-1}$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plots.plot_margins()
    plt.savefig('%s_powerden_%s_%s.pdf' % (starname, minfreq, maxfreq))

    # Save data in textfile
    print('Write %d entries to %s' % (len(freq), pds))
    timerStart = now()

    savenpz(pds, np.transpose([freq, powerden]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def background(nu_max):
    print('Corrects for granulation')
    """ This is based on background modelling due to granulation and
    described in 'Automated extraction of oscillation parameters for
    Kepler observations of solar-type stars' by Huber et al.
    """
    # Load data
    freq, powerden = loadnpz(pds).T

    P_n = np.mean(powerden[-1500:-1])
    guess_sigma_0 = np.sqrt(np.mean(powerden ** 2))
    guess_tau_0 = 1 / nu_max
    guess_sigma_1 = np.sqrt(np.mean(powerden ** 2))
    guess_tau_1 = 1 / nu_max

    # Eq. 1 in mentioned paper
    def background_fit_2(nu, sigma_0, tau_0):
        k1 = ((4 * sigma_0 ** 2 * tau_0) /
              (1 + (2 * np.pi * nu * tau_0) ** 2 +
               (2 * np.pi * nu * tau_0) ** 4))
        return k1

    def background_fit(nu, sigma_0, tau_0, sigma_1, tau_1):
        k1 = background_fit_2(nu, sigma_0, tau_0)
        k2 = background_fit_2(nu, sigma_1, tau_1)
        return P_n + k1 + k2

    def logbackground_fit(nu, sigma_0, tau_0, sigma_1, tau_1):
        return np.log10(background_fit(nu, sigma_0, tau_0, sigma_1, tau_1))

    # Cut out around the signals in order not to overfit them
    minimum = 600
    maximum = 1300

    filt = (freq > minimum) & (freq < maximum)
    freq_filt = freq[~filt]
    powerden_filt = powerden[~filt]

    # Fit
    z0 = [guess_sigma_0, guess_tau_0, guess_sigma_1, guess_tau_1]
    popt, pcov = curve_fit(logbackground_fit, freq_filt,
                           np.log10(powerden_filt), p0=z0)

    plt.figure()
    plt.loglog(freq, powerden, 'k', basex=10, basey=10, linewidth=0.1)
    plt.loglog(freq, background_fit(freq, *popt), 'r-', basex=10,
               basey=10)
    plt.loglog(freq, P_n + background_fit_2(freq, *popt[:2]), 'r--',
               basex=10, basey=10)
    plt.loglog(freq, P_n + background_fit_2(freq, *popt[2:]), 'r--',
               basex=10, basey=10)
    plt.loglog(freq, np.ones(len(freq))*P_n, 'r--')
    # plt.title(r'The power density spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\, \mu$Hz^{-1}$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plt.ylim([10 ** (-2), 2 * 10 ** (2)])
    plots.plot_margins()
    plt.savefig(r'%s_backgroundfit_%s_%s.pdf' % (starname,
                minfreq, maxfreq))

    # Correct for this simulated background by dividing it out
    corr_powerden = powerden / background_fit(freq, * popt)

    plt.figure()
    plt.loglog(freq, powerden, '0.75', basex=10, basey=10)
    plt.loglog(freq, corr_powerden, 'k', basex=10, basey=10)
    # plt.title(r'The corrected power density spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power density [ppm$^2\, \mu$Hz^{-1}$]')
    plt.xlim([np.amin(freq), np.amax(freq)])
    plots.plot_margins()
    plt.savefig(r'%s_backgroundcorrected_%s_%s.pdf' % (starname,
                minfreq, maxfreq))
    #plt.show()

    time, flux = loadnpz(ts).T

    # Change the amplitude by dividing the PDS with L
    L = time[-1]
    print('The length of the time series is %s Ms' % L)
    corr_power = corr_powerden / L

    plt.figure()
    plt.plot(freq, corr_power, 'k', linewidth=0.2)
    # plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.xlim([np.amin(freq), 2000])
    plots.plot_margins()
    plt.savefig(r'%s_powerspectrum_final_%s_%s.pdf' % (starname,
                minfreq, maxfreq))
    #plt.show()

    # Save data in textfile
    print('Write %d entries to %s' % (len(corr_power), cps))
    timerStart = now()

    savenpz(cps, np.transpose([freq, corr_power]))

    elapsedTime = now() - timerStart
    print('Took %.2f s' % elapsedTime)


def find_peaks(freq, power, minheight, comparorder):
    print('Find peaks')

    # Find relative maxima using scipy.signal.argrelmax
    point = scipy.signal.argrelmax(power, order=comparorder)
    if len(point[0]) == 0:
        raise Exception("No maxima found")

    # Find location and height of peaks
    peak = freq[point]
    height = power[point]

    # Only peaks over a given height 'minheight' should be included
    included_peak = peak[height > minheight]
    included_height = height[height > minheight]

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
    plots.plot_margins()
    plt.savefig(r'%s_nm%s_%s_%s.pdf' % (starname, nmsigma, minfreq,
                maxfreq))
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
        corr = acf(spower, nlags=lags)

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
    plt.plot(nautocorr, autocorr, 'k', linewidth=0.2)
    plt.plot(included_peak, included_height, 'b.')
    plt.plot(freqcut, heightcut, 'g', linewidth=0.1)
    plt.plot(freqcut, gauss(freqcut, *popt), 'b', linewidth=0.1)
    plt.plot(delta_nu, gauss(delta_nu, *popt), 'r.')
    plt.xlim([np.amin(nautocorr), 1000])
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Autocorrelated function')
    plt.ylim([0, 0.5])
    plots.plot_margins()
    plt.savefig(r'dn%s_%s_%s.pdf' % (starname, minfreq, maxfreq))

    # Find nu_max using the method found in
    # http://arxiv.org/pdf/0910.2764v1.pdf
    cutlimit = 3 * delta_nu
    overlap = 50
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
    plt.xlabel(r'Central frequency [$\mu$Hz]')
    plt.ylabel(r'Spacing [$\mu$Hz]')
    plt.xlim([freqlist[0, 0], freqlist[-1, 0]])
    plt.imshow(-clist.T, cmap='gray', interpolation='bilinear',
               origin='lower', aspect='auto',
               extent=[freqlist[0, 0], freqlist[-1, 0], 0, spacinglist[0, -1]])
    plt.savefig(r'%s_numax_huber_2_%s_%s.pdf' % (starname,
                cutlimit, overlap))

    # Collapse the ACF
    collacf = np.sum(clist, axis=1)
    xs = np.linspace(freqlist[0, 0], freqlist[-1, 0], 1000)
    popt = gaussian_fit(freqlist[:, 0], collacf)
    nu_max = popt[1]
    print(nu_max)

    fig = plt.figure()
    plt.xlabel(r'Central frequency [$\mu$Hz]')
    plt.ylabel(r'Collapsed ACF [Arbitary Units]')
    plt.xlim([freqlist[0, 0], freqlist[-1, 0]])
    plt.plot(freqlist[:, 0], collacf, 'k')
    plt.plot(xs, gauss(xs, *popt), 'b')
    plt.savefig(r'%s_numax_huber_3_%s_%s.pdf' % (starname,
                cutlimit, overlap))
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
    # T_eff = 6347
    # s_teff = 46
    # From the interferometric analysis
    T_eff = 6211
    s_teff = 91
    # From half the timeseries
    s_nmax = 21
    s_dn = 0.2

    # Find asteroseismic parameters
    delta_nu, nu_max = find_deltanu_and_nu_max()

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

    minimum = 500
    maximum = 1500

    filt = (freq > minimum) & (freq < maximum)
    freq = freq[filt]
    spower = spower[filt]

    peak, height = echelle(54.18, freq, spower)

    plt.figure()
    plt.plot(freq, spower, 'k', linewidth=1)
    #plt.plot(peak, height, 'ro')

    n, l, f = np.loadtxt('10005473fre.txt', skiprows=1, usecols=(0, 1, 2, )).T
    #timpeak = np.loadtxt('181096.pkb', usecols=(2,)).T
    #amaliepeak = np.loadtxt('181096.txt', usecols=(2,)).T
    plt.plot(f, np.ones(len(f)) * 2, 'bo')
    plt.plot(peak, np.ones(len(peak)) * 2.5, 'ro')
    """
    for freq in f:
        plt.axvline(x=freq, color='b', linestyle='-')
    for peak in amaliepeak:
        plt.axvline(x=peak, color='r', linestyle='-')
    """
    plt.xlim([minimum, maximum])
    # plt.title(r'The power spectrum of %s' % starname)
    plt.xlabel(r'Frequency [$\mu$Hz]')
    plt.ylabel(r'Power [ppm$^2$]')
    plt.savefig('zoom_cps%s_%s_%s_min_%smax_%s.pdf' %
                (starname, minfreq, maxfreq, minimum, maximum))
    plt.show()

    print(np.transpose([peak, np.round((peak/54.5)-1)]))


# Make an Échelle diagram
def echelle(delta_nu, freq, power):
    print('Plot Échelle diagram')

    e_comparorder = 700
    e_minheight = 0.17 * np.amax(power)

    peak, height = find_peaks(freq, power, e_minheight, e_comparorder)
    peakmod = np.mod(peak, delta_nu)

    n, l, f = np.loadtxt('10005473fre.txt', skiprows=1, usecols=(0, 1, 2, )).T
    #timpeak = np.loadtxt('181096.pkb', usecols=(2,)).T
    amaliepeak = np.loadtxt('181096.txt', usecols=(2,)).T

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


if __name__ == "__main__":
    # Here the functions are called
    make_the_timeseries()
    make_the_power_spectrum()
    smooth_power_spectrum()
    power_density_spectrum()
    background(nu_max_guess)
    scalingrelation()
    #plot_ps()
    #npzsavetxt(ts, ('%s/timeseries_%s.txt' % (direc, para)))
    #npzsavetxt(ps, ('%s/power_%s.txt' % (direc, para)))
