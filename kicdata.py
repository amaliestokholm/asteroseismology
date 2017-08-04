"""
This file defines a function to read and filter the Kepler data
"""

def fix_margins():
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)

def getdata(ID, kernelsize, quarter, sigma, noisecut):
    """
    This function returns the (time, flux)-data from the desired star.
    Arguments:
        - 'ID':             Choice of star
        - 'kernelsize':     The kernel-size for the median filter.
                            NB: must be an odd number.
        - 'quarters':       Chosen period of time
        - 'sigma':          Limitting sigma for sigma clipping.
        - 'noisecut':       Added for data sets with instrumental noise.
                            All data below the noisecut will be removed.
    """
    # Import modules
    import numpy as np
    import scipy.signal
    import os
    from time import time as now
    import matplotlib
    import plots

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
        matplotlib.rc('text.latex', preamble=
                      r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

    # matplotlib_setup()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Activate Seaborn color aliases
    sns.set_palette('colorblind')
    sns.set_color_codes(palette='colorblind')
    plt.style.use('ggplot')
    sns.set_context('poster')
    sns.set_style("ticks")
    
    def fix_margins():
        plots.plot_margins()
        #plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)

    # Find data files in path
    datafiles = sorted([s for s in os.listdir('data/%s/kepler' % ID)
                        if s.endswith('.dat')])
    # datafiles = datafiles[0:(int(quarter)+1)]

    # Starting time
    Q1 = np.loadtxt('./data/%s/kepler/%s' % (ID, datafiles[0]),
                    skiprows=8)
    t0 = Q1[0, 0]

    # Iterate over each quarter of Kepler data
    totaltime = []
    totalflux = []
    totaltime_noise = []
    totalflux_noise = []
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
        corr_time_sig = time[sigmaclip]
        corr_flux_sig = corr_flux[sigmaclip]
        print(' %s data points removed by sigma clipping'
              % np.sum(np.logical_not(sigmaclip)))

        print('After sigma cut, len=%s and %s' % (len(corr_time_sig),
                                                  len(corr_flux_sig)))

        # Extra filter in order to remove instrumental noise
        
        #diff = np.diff(corr_flux_sig)
        #diff = np.append(diff, [0])
        #assert diff.size == corr_flux_sig.size
        #diff_sigma = np.std(diff)
        #noiseclip = diff < (3 * diff_sigma)
        
        noiseclip = (corr_flux_sig > noisecut)
        corr_time_nos = corr_time_sig[noiseclip]
        corr_flux_nos = corr_flux_sig[noiseclip]
        print(' %s data points removed by noise clipping'
              % sum(np.logical_not(noiseclip)))

        print('After noise removal, len=%s and %s' % (len(corr_time_nos),
                                                      len(corr_flux_nos)))
        data_time = corr_time_nos
        data_flux = corr_flux_nos

        print('After zero removal, len = %s' % len(data_flux))

        # Write data to lists
        totaltime = np.r_[totaltime, time]
        totalflux = np.r_[totalflux, corr_flux]
        totaltime_noise = np.r_[totaltime_noise, corr_time_sig[~noiseclip]]
        totalflux_noise = np.r_[totalflux_noise, corr_flux_sig[~noiseclip]]
        totaldatatime = np.r_[totaldatatime, data_time]
        totaldataflux = np.r_[totaldataflux, data_flux]

        # Info-print
        print('%d/%d: %d data points remain after filtering'
              % (k+1, len(datafiles), len(totaldatatime)))

    elapsedTime = now() - timerStart
    print("Iteration over the quarters took %.2f s" % elapsedTime)

    # Plot the raw data
    plt.figure()
    # fix_margins()

    """ 
    The next step replaces datapoints in the most dense areas of the
    time series with a filled figure. This is only done in order to
    minimize the rendering time of the figure in the pdf file.
    This should only be used for plot optimisation.
    """
    """
    totaldatatime_norect = totaldatatime
    totaldataflux_norect = totaldataflux
    rects = [
        ((0.01, 2.65), (-0.5e-4, +0.5e-4)),
        ((2.76, 3.38), (-0.5e-4, +0.5e-4)),
        ((3.89, 5.33), (-0.5e-4, +0.5e-4)), 
    ]
    rect_points = 0
    for (x1, x2), (y1, y2) in rects:
        f = ((x1 <= totaldatatime_norect) & (totaldatatime_norect <= x2) &
             (y1 <= totaldataflux_norect) & (totaldataflux_norect <= y2))
        rect_points += f.sum()
        totaldatatime_norect = totaldatatime_norect[~f]
        totaldataflux_norect = totaldataflux_norect[~f]
        plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], 'k')
    print("%d/%d points coalesced to %d rectangles" %
          (rect_points, len(totaldatatime), len(rects)))
    plt.fill([np.amin(totaldatatime), np.amax(totaldatatime),
              np.amax(totaldatatime), np.amin(totaldatatime)],
             [noisecut, noisecut, -np.amax(totaldataflux),
              -np.amax(totaldataflux)], color='0.75')

    plt.plot(totaldatatime_norect, totaldataflux_norect,
             color='k', marker='.', ms=1, linestyle='None')
    """

    plt.plot(totaldatatime[::10], totaldataflux[::10], color='navy', marker='.', ms=5,
             linestyle='None')
    plt.plot(totaltime_noise[::10], totalflux_noise[::10],
             color='slategrey', marker='x', ms=5, linestyle='None', mew=1)
    plt.xlabel('Relative time [Ms]')
    plt.ylabel('Relative photometry')
    plt.xlim([np.amin(totaldatatime), np.amax(totaldatatime)])
    plt.ylim([-np.amax(totaldataflux), np.amax(totaldataflux)])
    # http://stackoverflow.com/a/17846471/1570972
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0), )
    plt.savefig('rawdata.pdf')

    return (totaldatatime, totaldataflux)
