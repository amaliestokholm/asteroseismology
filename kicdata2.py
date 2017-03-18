"""
This file define a function to read and filter the Kepler data
"""


def getdata(ID, kernelsize, quarter, sigma, noisecut):
    """
    This function returns the (time, flux)-data from the desired star.
    Arguments:
        - 'ID': Choice of star
        - 'kernelsize': The kernel-size for the median filter.
                        NB: must be an odd number.
        - 'quarters': Chosen period of time
        - 'sigma': Limitting sigma for sigma clipping.
    """
    # Import modules
    import numpy as np
    import scipy.signal
    import os
    from time import time as now
    import matplotlib

    def matplotlib_setup():
        """ The setup, which makes nice plots for the report"""
        fig_width_pt = 388
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

    # Find data files in path
    datafiles = sorted([s for s in os.listdir('data/%s/kepler' % ID)
                        if s.endswith('.dat')])
    datafiles = datafiles[0:(int(quarter)+1)]

    # Starting time
    Q1 = np.loadtxt('./data/%s/kepler/%s' % (ID, datafiles[0]),
                    skiprows=8)
    t0 = Q1[0, 0]

    # Iterate over each quarter of Kepler data
    totaltime = []
    totalflux = []
    totaldatatime = []
    totaldataflux = []

    timerStart = now()

    for (k, filename) in enumerate(datafiles):
        # Load the datafile and save the data in varibles
        keplerdata = np.loadtxt('./data/%s/kepler/%s' % (ID, filename),
                                skiprows=8)
        time = keplerdata[:, 0]
        flux = keplerdata[:, 1]

        # Convert time in truncated barycentric julian date to
        # relative time in mega seconds
        time -= t0
        time *= (60 * 60 * 24) / (1e6)

        # Remove invalid data (such as Inf or NaN)
        time = time[np.isfinite(flux)]
        flux = flux[np.isfinite(flux)]

        print('After Inf removal, len=%s' % len(flux))

        # Median-filtering (calculate the median and find the diff.)
        median = scipy.signal.medfilt(flux, kernelsize)
        corr_flux = np.divide(flux, median) - 1

        # Sigma clipping
        sigmaclip = (abs(corr_flux - np.mean(corr_flux)) <
                     sigma * np.std(corr_flux))
        corr_time_sig = time
        corr_time_sig[~sigmaclip] = 0
        corr_flux_sig = corr_flux
        corr_flux_sig[~sigmaclip] = 0
        print(' %s data points removed by sigma clipping'
              % np.sum(np.logical_not(sigmaclip)))

        print('After sigma cut, len=%s and %s' % (len(corr_time_sig),
                                                  len(corr_flux_sig)))

        # Extra filter in order to remove instrumental noise
        noiseclip = (corr_flux_sig > noisecut)
        corr_time_nos = corr_time_sig
        corr_time_nos[~noiseclip] = 0
        corr_flux_nos = corr_flux_sig
        corr_flux_nos[~noiseclip] = 0
        print(' %s data points removed by noise clipping'
              % sum(np.logical_not(noiseclip)))

        print('After noise removal, len=%s and %s' % (len(corr_time_nos),
                                                      len(corr_flux_nos)))

        # Remove the 'bad data' from the not-detrended data
        data_time = time[np.nonzero(corr_flux_nos)]
        data_flux = flux[np.nonzero(corr_flux_nos)]

        print('After zero removal, len = %s' % len(data_flux))

        # Write data to lists
        totaltime = np.r_[totaltime, time]
        totalflux = np.r_[totalflux, flux]
        totaldatatime = np.r_[totaldatatime, data_time]
        totaldataflux = np.r_[totaldataflux, data_flux]
        
        # Print info-print
        print('%d/%d: %d data points remain after filtering'
              % (k+1, len(datafiles), len(totaldatatime)))

    elapsedTime = now() - timerStart
    print("Iteration over the quarters took %.2f s" % elapsedTime)

    print('len = %s and %s, mean = %s' % (len(totaltime), len(totaldatatime), np.mean(totaltime)))

    # Plot the raw data
    rdfig = plt.figure()
    plt.plot(totaltime, totalflux,
             color='0.5', marker='.', linestyle='None')
    plt.plot(totaldatatime, totaldataflux,
             color='k', marker='.', linestyle='None')
    plt.title(r'The time series')
    plt.xlabel(r'Relative time [Ms]')
    plt.ylabel(r'Photometry')
    plt.xlim([np.amin(totaldatatime), np.amax(totaldatatime)])
    plt.savefig('rawdata.pdf')
    plt.show(rdfig)

    return (totaldatatime, totaldataflux)
