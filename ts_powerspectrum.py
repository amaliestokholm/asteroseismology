"""
This file defines a function to calculate the power spectrum of a star
"""
# Import modules
import numpy as np
from time import time as now
import os
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm


# Make nice plots
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


def power_spectrum(time, amplitude, weight=None, minfreq=None, maxfreq=None,
                   oversample=None, memory_use=None):
    """
    This function returns the power spectrum of the desired star.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'amplitude': Photometry data from the timeserie analysis.
        - 'weight': Weights for each point in the time series.
        - 'minfreq': The lower bound for the frequency interval
        - 'maxfreq': The upper bound for the frequency interval
        - 'oversample': The resolution of the power spectrum.
        - 'memory_use': The amount of memory used for this calculation.
    """
    # The default longest wavelength is the length of the time series.
    if minfreq is None:
        minfreq = 1 / (time[-1] - time[0])
    # The default greatest frequency is the Nyquist frequency.
    if maxfreq is None:
        maxfreq = 1 / (2 * np.median(np.diff(time)))
    # By default oversample 4 times
    if oversample is None:
        oversample = 4
    # By default use 500000 memory cells (8 bytes each).
    if memory_use is None:
        memory_use = 500000
    if weight is None:
        weight = np.ones(amplitude.shape)
    else:
        weight = np.asarray(weight)
        assert weight.shape == amplitude.shape

    # Generate cyclic frequencies
    step = 1 / (oversample * (time[-1] - time[0]))
    freq = np.arange(minfreq, maxfreq, step)

    # Generate list to store the calculated power
    alpha = np.zeros((len(freq),))
    beta = np.zeros((len(freq),))

    # Convert frequencies to angular frequencies
    nu = 2 * np.pi * freq

    # Iterate over the frequencies
    timerStart = now()

    # After this many frequencies, print progress info
    print_every = 75e6 // len(time)

    # Define a chunk for the calculation in order to save time
    chunksize = memory_use // len(time)
    chunksize = max(chunksize, 1)

    # Ensure chunksize divides print_every
    print_every = (print_every // chunksize) * chunksize

    for i in range(0, len(nu), chunksize):
        # Define chunk
        j = min(i + chunksize, len(nu))
        rows = j - i

        # Info-print
        if i % print_every == 0:
            elapsedTime = now() - timerStart
            if i == 0:
                totalTime = 0.004 * len(nu)
            else:
                totalTime = (elapsedTime / i) * len(nu)

            print("Progress: %.2f%% (%d of %d)  "
                  "Elapsed: %.2f s  Total: %.2f s"
                  % (np.divide(100.0*i, len(nu)), i, len(nu),
                     elapsedTime, totalTime))

        """
        The outer product is calculated. This way, the product between
        time and ang. freq. will be calculated elementwise; one column
        per frequency. This is done in order to save computing time.
        """
        nutime = np.outer(time, nu[i:j])

        """
        An array with the measured amplitude is made so it has the same size
        as "nutime", since we want to multiply the two.
        """
        amplituderep = amplitude.reshape(-1, 1)
        weightrep = weight.reshape(-1, 1)

        # The Fourier subroutine
        sin_nutime = np.sin(nutime)
        cos_nutime = np.cos(nutime)

        s = np.sum(weightrep * sin_nutime * amplituderep, axis=0)
        c = np.sum(weightrep * cos_nutime * amplituderep, axis=0)
        ss = np.sum(weightrep * sin_nutime ** 2, axis=0)
        cc = np.sum(weightrep * cos_nutime ** 2, axis=0)
        sc = np.sum(weightrep * sin_nutime * cos_nutime, axis=0)

        alpha[i:j] = ((s * cc) - (c * sc)) / ((ss * cc) - (sc ** 2))
        beta[i:j] = ((c * ss) - (s * sc)) / ((ss * cc) - (sc ** 2))

    alpha = alpha.reshape(-1, 1)
    beta = beta.reshape(-1, 1)
    freq = freq.reshape(-1, 1)
    power = alpha ** 2 + beta ** 2
    elapsedTime = now() - timerStart
    print('Computed power spectrum in %.2f s' % (elapsedTime))
    return (freq, power, alpha, beta)


def daystomegaseconds(time):
    """
    Convert time in truncated barycentric julian date to relative time
    in mega seconds
    """
    time -= time[0]
    time *= (60 * 60 * 24) / (1e6)
    return time


def spectralwindow(f, time, **kwargs):
    """
    This function returns the spectral window of a time series.
    Arguments:
        - 'f': A given frequency.
        - 'time': Photometry data from the timeserie analysis.
        - '**kwargs': Keyword arguments for the power calculation.
    """
    nutime = 2 * np.pi * f * time
    acos = np.cos(nutime)
    freq, pcos, alpha, beta = power_spectrum(time, acos, **kwargs)
    asin = np.sin(nutime)
    freq, psin, alpha, beta = power_spectrum(time, asin, **kwargs)
    sw = 0.5 * (pcos + psin)
    return freq, sw


def window(time, alpha, beta, f, memory_use=500000):
    """
    This function returns a generated time series.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'alpha': The alpha term for the power calculation
        - 'beta': The beta term for the power calculation
        - 'f': The frequency (or frequencies) of the time series.
        - 'memory_use': The amount of memory used for this calculation.
    """
    alpha = np.asarray(alpha)
    k = alpha.shape[0]
    assert alpha.shape == (k, 1)
    assert beta.shape == (k, 1)
    assert f.shape == (k, 1)

    # Generate list to store the calculated window
    w = np.zeros((len(time),))

    # Define a chunk for the calculation in order to save time
    chunksize = memory_use // len(time)
    chunksize = max(chunksize, 1)

    nu = 2 * np.pi * f
    for i in range(0, len(nu), chunksize):
        # Define chunk as in power_spectrum
        m = np.zeros((len(time), 1))
        j = min(i + chunksize, len(nu))
        rows = j - i

        nutime = np.outer(time, nu[i:j])
        sin_nutime = np.sin(nutime)
        cos_nutime = np.cos(nutime)
        m = (np.transpose(alpha[i:j]) * sin_nutime +
             np.transpose(beta[i:j]) * cos_nutime)

        assert w.shape == time.shape
        assert m.shape[0] == w.shape[0]
        w += m.sum(axis=1)
    assert w.shape == time.shape
    return w


def CLEAN(time, amplitude, k, **kwargs):
    """
    This function uses the iterative algorithm CLEAN to clean out the
    highest frequency including its sidelobes one of a time.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'amplitude': Amplitude from the timeserie analysis.
        - 'k': The number of peaks which will be removed.
        - '**kwargs': Keyword arguments for the power calculation.
                      The oversampling can thus be changed to more
                      precisely determine the frequencies of the
                      highest peaks.
    """
    osc = []
    for i in range(k):
        print(i)
        freq, power, alpha, beta = power_spectrum(time, amplitude, **kwargs)
        f = np.argmax(power)
        w = window(time, alpha[f].reshape(-1, 1),
                   beta[f].reshape(-1, 1), freq[f].reshape(-1, 1))
        amplitude = amplitude - w
        osc.append((freq[f], power[f], alpha[f], beta[f]))
    return time, amplitude, osc


def lowpass(time, alpha, beta, freq, cutoff):
    """
    This function makes a low-pass filter.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'alpha': The alpha term for the power calculation
        - 'beta': The beta term for the power calculation
        - 'freq': The frequencies of the time series.
        - 'cutoff': The frequency above which no power is passed
                    through the filter.
    """
    k = alpha.shape[0]
    assert alpha.shape == (k, 1)
    assert beta.shape == (k, 1)
    assert freq.shape == (k, 1)

    freqfilt = freq < cutoff
    lpfreq = freq[freqfilt].reshape(-1, 1)
    alpha = alpha[freqfilt].reshape(-1, 1)
    beta = beta[freqfilt].reshape(-1, 1)

    T = freq[-1] - freq[0]
    f = T/2

    num = window(time, alpha, beta, lpfreq)
    swfreq, sw = spectralwindow(f, time)
    den = np.sum(sw)
    lp = num/den
    return lp


def highpass(time, amplitude, alpha, beta, freq, cutoff):
    """
    This function makes a high-pass filter.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'amplitude': Amplitude from the time serie analysis.
        - 'alpha': The alpha term for the power calculation
        - 'beta': The beta term for the power calculation
        - 'freq': The frequencies of the time series.
        - 'cutoff': The frequency below which no power is passed
                    through the filter.
    """
    lp = lowpass(time, alpha, beta, freq, cutoff)
    hp = amplitude - lp
    return hp


def bandpass(time, alpha, beta, freq, minfreq, maxfreq):
    """
    This function makes a band-pass filter.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'alpha': The alpha term for the power calculation
        - 'beta': The beta term for the power calculation
        - 'freq': The frequencies of the time series.
        - 'minfreq': The lower frequency limit below which no power is passed
                    through the filter.
        - 'maxfreq': The upper frequency limit above which no power is passed
                    through the filter.
    """
    k = alpha.shape[0]
    assert alpha.shape == (k, 1)
    assert beta.shape == (k, 1)
    assert freq.shape == (k, 1)

    freqfilt = (minfreq < freq) & (freq < maxfreq)
    lpfreq = freq[freqfilt].reshape(-1, 1)
    alpha = alpha[freqfilt].reshape(-1, 1)
    beta = beta[freqfilt].reshape(-1, 1)

    T = freq[-1] - freq[0]
    f = T/2

    num = window(time, alpha, beta, lpfreq)
    swfreq, sw = spectralwindow(f, time)
    den = np.sum(sw)
    bp = num/den
    return bp


def running_mean(x, N):
    """
    This calculates the running mean using the cumulative sum in a
    specified window.
    (see wikipedia -> Moving average -> Cumulative moving average.)
    Arguments:
        - 'x': Data series
        - 'N': The size of the window

    The output has N-1 fewer entries than x.

    >>> running_mean([0, 1, 2, 3, 4], 2)
    array([ 0.5,  1.5,  2.5,  3.5])
    """
    x = np.asarray(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def running_median(x, N):
    """
    >>> running_median([0, 1, 2, 3, 4], 3)
    array([1, 2, 3])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    filt = scipy.signal.medfilt(x, N)
    margin = (N - 1) // 2
    return filt[margin:-margin]


def running_mean_filter(x, N):
    """
    This fixes the length of the running mean so it has the same length
    as the inputted data series by repeating the last entry $N-1$ times.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.

    >>> running_mean_filter([0, 1, 2, 3, 4], 3)
    array([ 1.,  1.,  2.,  3.,  3.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    y = running_mean(x, N)
    extra_count = (N - 1) / 2
    left = np.repeat(y[0], extra_count)
    right = np.repeat(y[-1], extra_count)
    return np.concatenate([left, y, right])


def running_mean_filter_2(x, N):
    """
    This fixes the length of the running mean so it has the same length
    as the inputted data series by repeating the last entry $N-1$ times.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.

    >>> running_mean_filter_2([0, 3, 9, 15, 18], 5)
    array([  0.,   4.,   9.,  14.,  18.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    for i in range(extra_count):
        left.append(np.mean(x[:2*i+1]))
        right.append(np.mean(x[-2*i-1:]))
    right.reverse()
    y = running_mean(x, N)
    return np.concatenate([left, y, right])


def running_median_filter(x, N):
    """
    >>> running_median_filter([0, 3, 9, 18, 24], 5)
    array([  0.,   3.,   9.,  18.,  24.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    for i in range(extra_count):
        left.append(np.median(x[:2*i+1]))
        right.append(np.median(x[-2*i-1:]))
    right.reverse()
    y = running_median(x, N)
    return np.concatenate([left, y, right])


def running_var_filter(x, N):
    """
    This calculates the variance of a given data series x
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.
    """
    y = running_mean_filter(x, N)
    var = running_mean_filter(x ** 2, N) - y ** 2
    return var


def weights(x, N):
    """
    This calculates the statistical weights of a given data series x.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.
    """
    var = running_var_filter(x, N)
    return 1 / np.maximum(1, var)


"""
The following functions are only used for plotting and for calling the
functions above
"""


def statistical_weights(filename, psfile, cutoff, N):
    time, amplitude = np.loadtxt(filename, usecols=(0, 1)).T
    time = daystomegaseconds(time)
    freq, power = np.loadtxt(psfile, usecols=(0, 1)).T
    hp = run_filter(filename, 'highpass', cutoff)
    (root, ext) = os.path.splitext(filename)
    hpfile = root + 'hp_%s.txt' % cutoff
    np.savetxt(hpfile, hp)
    weight = weights(hp, N)
    freq, power, alpha, beta = power_spectrum(time, amplitude, weight=weight)
    (root, ext) = os.path.splitext(filename)
    psfile = 'weightedps_%s_cutoff%s_N%s.txt' % (root, cutoff, N)
    np.savetxt(psfile, np.transpose([freq.reshape(-1), power.reshape(-1)]))
    plot_powerspectrum(psfile)


def loadnpz(filename):
    return np.load(filename)['data']


def run_ps(filename, minfreq=None, maxfreq=None, oversample=None,
           memory_use=None):
    A = np.loadtxt(filename).T
    time = A[0]
    amplitude = A[1]
    weight = A[3]
    print(weight)

    time = daystomegaseconds(time)

    # Calculate and print Nyquist-frequency
    nyquist = np.divide(1, 2 * np.median(np.diff(time)))
    print('Nyquist frequency: %s µHz' % str(nyquist))

    (freq, power, alpha, beta) = power_spectrum(time, amplitude, weight,
                                                minfreq, maxfreq,
                                                oversample, memory_use)
    (root, ext) = os.path.splitext(filename)
    psfile = 'ps_%s_min%s_max%s_os%s.txt' % (root, freq[0], freq[-1],
                                             oversample)
    np.savetxt(psfile, np.transpose([freq.reshape(-1), power.reshape(-1)]))
    plot_powerspectrum(psfile)


def run_CLEAN(filename, k, **kwargs):
    initime, iniamplitude = np.loadtxt(filename).T
    initime = daystomegaseconds(initime)
    #initime, iniamplitude = loadnpz(filename).T
    inifreq, inipower, inialpha, inibeta = power_spectrum(initime,
                                                          iniamplitude,
                                                          **kwargs)
    time, amplitude, osc = CLEAN(initime, iniamplitude, k, **kwargs)
    endfreq, endpower, endalpha, endbeta = power_spectrum(time,
                                                          amplitude,
                                                          **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Frequency [$\mu$Hz]')
    ax.set_ylabel(r'Power [ppm$^2$]')
    ax.set_title(r'The CLEANed power spectrum')
    ax.plot(inifreq, inipower, 'k')
    ax.plot(endfreq, endpower, 'r')
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    (root, ext) = os.path.splitext(filename)
    psfile = 'clean_%s_k%s_min%s_max%s.txt' % (root, k,
                                               np.amin(endfreq),
                                               np.amax(endfreq))
    peakfile = 'cleanpeaks_%s_k%s_min%s_max%s.txt' % (root, k,
                                                      np.amin(endfreq),
                                                      np.amax(endfreq))
    peakfile2 = 'clean_timamp_%s_k%s_min%s_max%s.txt' % (root, k,
                                                         np.amin(endfreq),
                                                         np.amax(endfreq))
    np.savetxt(peakfile, osc)
    np.savetxt(peakfile2,
               np.transpose([time.reshape(-1), amplitude.reshape(-1)]))
    np.savetxt(psfile,
               np.transpose([endfreq.reshape(-1), endpower.reshape(-1)]))
    fig.savefig('clean_%s_k%s_min%s_max%s.pdf' % (root, k,
                                                  np.amin(endfreq),
                                                  np.amax(endfreq)))
    return time, amplitude


def run_bandpass(filename, minfreq, maxfreq, **kwargs):
    time, amplitude = np.loadtxt(filename).T
    time = daystomegaseconds(time)
    amplitude = amplitude - np.mean(amplitude)
    freq, power, alpha, beta = power_spectrum(time, amplitude, **kwargs)
    bp = bandpass(time, alpha, beta, freq, minfreq, maxfreq)
    bpfreq, bppower, bpalpha, bpbeta = power_spectrum(time, bp, **kwargs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Time [Ms]')
    ax.set_ylabel(r'Amplitude ')
    ax.set_title(r'Band-pass filter')
    ax.plot(time, amplitude, 'r', linewidth=0.5)
    ax.plot(time, bp, 'b')
    ax.set_xlim([time[0], time[-1]])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    (root, ext) = os.path.splitext(filename)
    bpfile = 'bp_%s_min%s_max%s.txt' % (root, minfreq, maxfreq)
    np.savetxt(bpfile, np.transpose([time.reshape(-1), bp.reshape(-1)]))
    fig.savefig(root + 'bp_%s_min%s_max%s.pdf' % (root, minfreq, maxfreq))
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Frequency [$\mu$Hz]')
    ax.set_ylabel(r'Power [ppm$^2$]')
    ax.set_title(r'Band-pass filter')
    ax.plot(freq, power, 'r', linewidth=0.5)
    ax.plot(bpfreq, bppower, 'b')
    ax.set_xlim([minfreq - 10, maxfreq + 10])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    (root, ext) = os.path.splitext(filename)
    bpfile = 'bppower_%s_min%s_max%s.txt' % (root, minfreq, maxfreq)
    np.savetxt(bpfile, np.transpose([bpfreq.reshape(-1), bppower.reshape(-1)]))
    fig.savefig(root + 'bppower_%s_min%s_max%s.pdf' % (root, minfreq, maxfreq))


def run_filter(filename, passfilter, cutoff, **kwargs):
    time, amplitude = np.loadtxt(filename, usecols=(0, 1)).T
    time = daystomegaseconds(time)

    amplitude = amplitude - np.mean(amplitude)
    if passfilter is not 'lowpass' and passfilter is not 'highpass':
        raise ValueError('wrong filtertype')
    freq, power, alpha, beta = power_spectrum(time, amplitude, **kwargs)
    if passfilter is 'lowpass':
        p = lowpass(time, alpha, beta, freq, cutoff)
        p = p + np.mean(amplitude)
    if passfilter is 'highpass':
        p = highpass(time, amplitude, alpha, beta, freq, cutoff)
        p = p + np.mean(amplitude)
    return p


def plot_file(filenames, xlabel='', ylabel='', title='',
              xlim=None, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    color = ['k', 'r', 'b', 'g']
    print(filenames)
    for filename, i in zip(filenames, np.arange(len(filenames))):
        print(filename)
        #x, y = loadnpz(filename).T
        x, y = np.loadtxt(filename).T
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        elif xlim is None:
            ax.set_xlim([x[0], x[-1]])
        plt.plot(x, y, color[i])
        (root, ext) = os.path.splitext(filename)
    fig.savefig(root + '.pdf')
    plt.show()


def plot_timeseries(filename, **kwargs):
    plot_file(filename, xlabel=r'Time [Ms]',
              ylabel=r'Amplitude [ppm]',
              title=r'The time series', **kwargs)


def plot_powerspectrum(filename, **kwargs):
    plot_file(filename, xlabel=r'Frequency [$\mu$Hz]',
              ylabel=r'Power [ppm$^2$]',
              title=r'The power spectrum', **kwargs)


def plot_spectralwindow(ts_filename, ps_filename, weighti=None, df=35,
                        weights=None, **kwargs):
    # end is used for debugging. If end=None all data points are used.
    end = None

    A = np.loadtxt(ts_filename).T
    A = A[:, :end]
    time = A[0]
    amplitude = A[1]

    time = daystomegaseconds(time)

    # The column of weights is in the data file and are defined here.
    if weighti is not None:
        weight = A[weighti]
    if weights is not None:
        weight = weights
    else:
        weight = None

    B = np.loadtxt(ps_filename).T
    freq = B[0]
    power = B[1]

    # The spectral window is calculated
    T = freq[-1] - freq[0]
    f = T/2

    swfreq, sw = spectralwindow(f, time, weight=weight, **kwargs)
    swfreq2, sw2 = spectralwindow(f, time, **kwargs)

    print("L2 difference: %s" % ((sw - sw2)**2).sum())
    assert (swfreq == swfreq2).all()

    minfreq = swfreq[np.argmax(sw)] - df
    maxfreq = swfreq[np.argmax(sw)] + df

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Frequency [$\mu$Hz]')
    ax2.set_ylabel(r'Power [ppm$^2$]')
    ax2.set_title(r'The spectral window')
    ax2.set_xlim([minfreq, maxfreq])
    plt.plot(swfreq, sw, 'k')
    plt.plot(swfreq2, sw2, 'r')
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    (root, ext) = os.path.splitext(ps_filename)
    fig.savefig(root + 'weight_spectral.pdf')
    np.savetxt(root + 'weight_spectral.txt',
               np.transpose([swfreq.reshape(-1), sw.reshape(-1)]))
    plt.show()


def plot_comparison(filenames, oscfiles, timampfiles):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Time [Ms]')
    ax.set_ylabel(r'Amplitude')
    ax.set_title(r'Band-pass filter')
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    #ax.set_ylim([-0.8, 0.8])
    i = 0
    j = i
    color = ['b', 'r', 'g', 'c']

    for filename in filenames:
        bptime, bp = np.loadtxt(filename).T
        plt.plot(bptime, bp, color[i])
        ax.set_xlim([bptime[0], bptime[-1]])
        i = i + 1

    bpfile = 'bandpasscompare.pdf'
    fig.savefig(bpfile)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    #ax2.set_ylim([-0.8, 0.8])
    ax2.set_xlabel(r'Time [Ms]')
    ax2.set_ylabel(r'Amplitude')
    ax2.set_title(r'Using CLEAN as a filter')

    for (oscfile, timampfile) in zip(oscfiles, timampfiles):
        freq, power, alpha, beta = np.loadtxt(oscfile).T
        time, amplitude = np.loadtxt(timampfile).T
        alpha = alpha.reshape(-1, 1)
        beta = beta.reshape(-1, 1)
        freq = freq.reshape(-1, 1)
        cleanwindow = window(time, alpha, beta, freq)
        plt.plot(time, cleanwindow, color[j])
        ax2.set_xlim([bptime[0], bptime[-1]])
        j = j + 1

    bpfile = 'cleancompare.pdf'
    fig2.savefig(bpfile)

    plt.show()


def running_mean(x, N):
    """
    This calculates the running mean using the cumulative sum in a
    specified window.
    (see wikipedia -> Moving average -> Cumulative moving average.)
    Arguments:
        - 'x': Data series
        - 'N': The size of the window

    The output has N-1 fewer entries than x.

    >>> running_mean([0, 1, 2, 3, 4], 2)
    array([ 0.5,  1.5,  2.5,  3.5])
    """
    x = np.asarray(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def running_median(x, N):
    """
    >>> running_median([0, 1, 2, 3, 4], 3)
    array([1, 2, 3])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    filt = scipy.signal.medfilt(x, N)
    margin = (N - 1) // 2
    return filt[margin:-margin]


def running_mean_filter(x, N):
    """
    This fixes the length of the running mean so it has the same length
    as the inputted data series by repeating the last entry $N-1$ times.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.

    >>> running_mean_filter([0, 1, 2, 3, 4], 3)
    array([ 1.,  1.,  2.,  3.,  3.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    y = running_mean(x, N)
    extra_count = (N - 1) / 2
    left = np.repeat(y[0], extra_count)
    right = np.repeat(y[-1], extra_count)
    return np.concatenate([left, y, right])


def running_mean_filter_2(x, N):
    """
    This fixes the length of the running mean so it has the same length
    as the inputted data series by repeating the last entry $N-1$ times.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.

    >>> running_mean_filter_2([0, 3, 9, 15, 18], 5)
    array([  0.,   4.,   9.,  14.,  18.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    for i in range(extra_count):
        left.append(np.mean(x[:2*i+1]))
        right.append(np.mean(x[-2*i-1:]))
    right.reverse()
    y = running_mean(x, N)
    return np.concatenate([left, y, right])


def running_median_filter(x, N):
    """
    >>> running_median_filter([0, 3, 9, 18, 24], 5)
    array([  0.,   3.,   9.,  18.,  24.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    for i in range(extra_count):
        left.append(np.median(x[:2*i+1]))
        right.append(np.median(x[-2*i-1:]))
    right.reverse()
    y = running_median(x, N)
    return np.concatenate([left, y, right])


def noiseremoval(time, flux, kernelsize=3, sigma=4):
    """
    This removes bad data.
    Arguments:
        - 'time': Time from the timeserie analysis.
        - 'flux': Flux from the timeseries analysis
        - 'kernelsize': The size of the window
    """

    # Median-filtering
    median = scipy.signal.medfilt(flux, kernelsize)
    dif = median / flux - 1
    std = sigma * np.std(dif) * np.ones(len(flux))

    # Sigma clipping
    corrflux = np.ones(len(flux))
    change = 0
    for i in np.arange(len(flux)):
        #if (abs(flux[i] - median[i]) > std[i]):
        if (abs(dif[i]) > std[i]):
            corrflux[i] = median[i]
            change += 1
        else:
            corrflux[i] = flux[i]
    print('noise removal changes = %s' % change)

    """
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [Ms]')
    ax2.set_ylabel(r'Amplitude')
    ax2.set_xlim([time[0], time[-1]])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    plt.plot(time, flux, 'k')
    plt.plot(time, corrflux, 'r')
    # fig.savefig('noiseremoval.pdf')

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [Ms]')
    ax2.set_ylabel(r'Amplitude')
    plt.plot(time, dif, 'k')
    plt.plot(time, std, 'r-')
    plt.plot(time, -std, 'r-')
    plt.show()
    """
    return time, corrflux


def normalise(time, flux, kernelsize=25):
    """
    This normalises the time series by dividing out the running mean
    of the running median from the time series.
    Arguments:
        - 'time': Time from the timeserie analysis.
        - 'flux': Flux from the timeseries analysis
        - 'kernelsize': The size of the window
    """
    median = running_median_filter(flux, kernelsize)
    mean = running_mean_filter_2(median, kernelsize)
    corrflux = flux / mean

    """
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [days]')
    ax2.set_ylabel(r'Amplitude')
    plt.plot(time, flux, 'k')
    plt.plot(time, median, 'r')
    plt.plot(time, mean, 'b')
    ax2.set_xlim([time[0], time[len(time)/5]])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    ax2.set_ylim([np.amin(flux), np.amax(flux)])
    #fig.savefig('normer.pdf')
    """

    return time, corrflux


"""
def step(x, dT, P, phi):
    pi = np.zeros(len(x))
    xs = np.floor(x[-1] / P)
    # assert P * xs <= x[-1] < P * (xs + 1)
    for m in np.arange(xs + 1):
        filt = np.abs(x - m * P - phi) < dT/2
        pi[filt] = 1
    return pi
"""


def step(x, dT, P, phi):
    """
    This produces the toy model for up-side down transits.
    Arguments:
        - 'x': Time from the timeserie analysis.
        - 'dT': The length of the transit.
        - 'P': The period of the transit.
        - 'phi': The phase of the transit.
    """
    x = x - phi
    a = x % P
    pi = np.zeros(len(x))
    pi[a < dT/2] = 1
    pi[a > P - dT/2] = 1
    return pi


def crosscorr(a, b):
    """
    This calculates the cross correlation coefficient between two
    datasets.
    Arguments:
        - 'a': A dataset
        - 'b': A dataset.
    """
    r = np.sum((a - np.mean(a)) * (b - np.mean(b)))
    asum = np.sum((a - np.mean(a)) ** 2)
    bsum = np.sum((b - np.mean(b)) ** 2)
    sqr = np.sqrt(asum * bsum)
    if sqr == 0:
        print('zero')
        sqr = 0.00001
    corr = r / sqr
    return corr


def splitchunks(time, flux):
    """
    This splits the time series into chunks outlined by detected gaps
    in the time series.
    Arguments:
        - 'time': Time from the timeserie analysis.
        - 'flux': Flux from the timeseries analysis
    """
    """
    >>> time = [1, 2, 3, 8, 9, 11, 20, 21]
    >>> np.median(np.diff(time))
    1.0
    >>> flux = [2, 2, 2, 2, 2, 2, 2, 2]
    >>> chunks = splitchunks(time, flux)
    >>> len(chunks)
    3
    >>> chunktime, chunkflux = zip(*chunks)
    >>> np.all(np.concatenate(chunktime) == time)
    True
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    difftime = np.diff(time)
    meddiff = np.median(difftime)
    chunks = []
    gap = difftime > 3 * meddiff
    chunkstart = 0
    for i in gap.nonzero()[0]:
        # The next chunk is indicies chunkstart to i
        chunks.append((time[chunkstart:i+1], flux[chunkstart:i+1],))
        chunkstart = i+1
    chunks.append((time[chunkstart:], flux[chunkstart:],))
    return chunks


def process_ts(filename, clean=None):
    """
    This runs the noiseremoval and normalisation for a given datafile.
    Arguments:
        - 'filename': A Kepler datafile for a star.
        - 'clean': Whether or not it should remove the most periodic
                   signal in the time series.
    """
    data = np.loadtxt(filename)
    time = data[:, 0]
    flux = data[:, 1]
    (root, ext) = os.path.splitext(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, flux)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    ax.set_xlim([time[0], time[-1]])
    ax.set_xlim([756, 770])
    fig.savefig(root + '_timeseries_test.pdf')
    plt.show()

    if clean is not None:
        print('Run CLEAN')
        time, flux, _osc = CLEAN(time, flux, 1)

    times = []
    fluxs = []
    chunks = splitchunks(time, flux)
    print('Split into %s chunks' % len(chunks))
    for time, flux in chunks:
        time, flux = noiseremoval(time, flux)
        time, flux = normalise(time, flux)
        times.append(time)
        fluxs.append(flux)

    time = np.concatenate(times)
    flux = np.concatenate(fluxs)

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [days]')
    ax2.set_ylabel(r'Amplitude')
    ax2.set_xlim([time[0], time[-1]])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    plt.plot(time, flux, 'k')
    fig.savefig(root + '_corrflux.pdf')
    plt.show()

    return time, flux


def matchfilter(filename, dT, P, phi, res, clean=None):
    """
    This calculates the cross correlation of a datafile
    Arguments:
        - 'filename': a Kepler datafile.
        - 'dT': The length of a transit
        - 'P': The period of a transit
        - 'phi': The phase of a transit
        - 'res': The resolution of the parameter space of (P, phi).
        - 'clean': Whether or not it should remove the most periodic
                   signal in the time series.
    """
    time, flux = process_ts(filename, clean)
    (root, ext) = os.path.splitext(filename)
    C = np.zeros(shape=(len(P), len(phi)))
    data = (1-flux)
    for i in np.arange(len(P)):
        if i % 10 == 0:
            print("%s/%s" % (i, len(P)))
        for j in np.arange(len(phi)):
            pi = step(time, dT, P[i], phi[j])
            cc = crosscorr(pi, data)
            C[i, j] = cc

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(r'Period [d]')
    ax.set_ylabel(r'Phase [d]')
    ax.set_zlabel(r'Correlation')
    Pmesh, phimesh = np.meshgrid(P, phi)
    ax.plot_surface(Pmesh, phimesh, C, rstride=10, cstride=10,
                    color='c', linewidth=0.01, antialiased=False)
    ax.set_zlim([-0.1, 0.2])
    fig.savefig(root + '_mesh%s.pdf' % dT)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Time [d]')
    ax.set_ylabel(r'1-flux')
    ax.plot(time, data, 'k')
    (k, l) = np.unravel_index(np.argsort(-C.ravel()), C.shape)
    colors = ['r', 'g', 'b', 'c']
    drawn = []
    planetfile = root + '_planets_%s.txt' % res
    fp = open(planetfile, "w")
    print("dT = %s" % dT, file=fp)
    for i in range(len(k)):
        if len(drawn) == len(colors):
            break
        if any((k[i]-k[j])**2 + (l[i]-l[j])**2 < 4**2 for j in drawn):
            continue
        c = colors[len(drawn)]
        drawn.append(i)
        print("%d (color %s): %f, %f (value %g)" %
              (len(drawn), c, P[k[i]], phi[l[i]], C[k[i], l[i]]))
        bestpi = step(time, dT, P[k[i]], phi[l[i]])
        print("Color %s: (%s, %s) value %g" %
              (c, P[k[i]], phi[l[i]], C[k[i], l[i]]), file=fp)
        ax.plot(time, bestpi, c)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([np.amin(data), np.amax(data)+0.001])

    fig.savefig(root + '_bestpi%s.pdf' % dT)
    plt.show()


def autocorr(filename, clean=None):
    """
    This calculates the autocorrelation of a datafile
    Arguments:
        - 'filename': a Kepler datafile.
        - 'clean': Whether or not it should remove the most periodic
                   signal in the time series.
    """
    time, flux = process_ts(filename, clean)
    (root, ext) = os.path.splitext(filename)

    data = 1 - flux

    # This creates a uniform time grid
    step = np.median(np.diff(time))
    start = time[0] - (0.5 * step)

    lag = np.floor((time - start) / step)
    dist = (time - start) / step - lag

    ss = np.mean((dist - 0.5) ** 2)
    print("Grid score of (%s, %s): %s" % (start, step, ss))
    print(np.amin(dist), np.amax(dist), np.median(dist))
    print(len(lag), len(np.unique(lag)))

    # This optimises the grid
    start = start + (np.median(dist)-0.5) * step

    lag = np.floor((time - start) / step)
    dist = (time - start) / step - lag

    ss = np.mean((dist - 0.5) ** 2)
    print("Grid score of (%s, %s): %s" % (start, step, ss))
    print(np.amin(dist), np.amax(dist), np.median(dist))
    assert len(lag) == len(np.unique(lag))

    # Autocorrelation
    lag = lag.astype(np.int64)
    a = np.zeros((np.amax(lag) + 1,))
    a[lag] = data

    N = len(a)
    a = (np.correlate(a, a, "full")
         / (float(N)-np.abs(np.arange(-N+1, N))))
    assert len(a) == 2*N-1
    a = a[len(a)//2:]
    assert len(a) == N
    steps = np.arange(len(a)) * step

    comparorder = 400
    # Find relative maxima using scipy.signal.argrelmax
    point = scipy.signal.argrelmax(a, order=comparorder)
    if len(point[0]) == 0:
        raise Exception("No maxima found")

    # Find location and height of peaks
    peak = steps[point]
    height = a[point]

    # Find a mean value
    print(peak[:10])
    #peak = peak[height < 0.5]
    bestP = np.mean(peak[:5] / (1+np.arange(5)))
    print(bestP, np.std(peak[:5] / (1+np.arange(5))))
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [days]')
    ax2.set_ylabel(r'Autocorrelation')
    ax2.set_xlim([np.amin(steps), 50])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    plt.plot(steps, a)
    plt.plot(peak, height, 'ro')
    fig.savefig(root + '_autocorr.pdf')
    plt.show()

    phasefold(time, flux, 0.1, bestP, phi=(0))


def phasefold(time, flux, dT, P, phi):
    """
    This makes a phase folding diagram
    Arguments:
        - 'time': The time from the time series
        - 'flux': The flux from the time series.
        - 'dT': The length of a transit
        - 'P': The period of a transit
        - 'phi': The phase of a transit
    """
    timep = (time - phi) % P
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [days]')
    ax2.set_ylabel(r'Amplitude')
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    #ax2.set_xlim([np.amin(timep), 1])
    plt.plot(timep, flux, 'k.')
    #plt.plot(timep, (-0.002*step(time, dT, P, phi))+1, 'r')
    #fig.savefig('phasefold_4.pdf')
    plt.show()

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'Time [days]')
    ax2.set_ylabel(r'Amplitude')
    ax2.set_xlim([time[0], time[-1]])
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    ax2.set_xlim([756, 770])
    ax2.set_xlim([np.amin(time), np.amin(time)+20])
    ax2.set_ylim(0.999, 1.001)
    plt.plot(time, 2-flux, 'k')
    plt.plot(time, step(time, dT, P, phi), 'r')
    #plt.plot(time, step(time, dT, bestPCC, phi), 'b')
    #fig.savefig('sol_4.pdf')
    plt.show()

if __name__ == "__main__":
    filename = 'KEP02.txt'

    dT = 0.1
    res = 200
    P = np.linspace(1, 10, res)  # By eye: 0.38
    phi = np.linspace(1, 10, res)  # By eye: 0.2
    #matchfilter(filename, dT, P, phi, res)
    autocorr(filename)
    """
    dT = 0.01/2  # np.linspace(0.01-0.001, 0.01-0.001, 5)
    P = np.linspace(0.38-0.2, 0.38+0.2, 500)  # By eye: 0.38
    phi = np.linspace(0.01, 0.3, 500)  # By eye: 0.2
    matchfilter(filename, dT, P, phi)
    """
