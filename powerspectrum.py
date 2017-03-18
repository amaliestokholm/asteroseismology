"""
This file defines a function to calculate the power spectrum of a star
"""


def power(time, flux, minfreq, maxfreq, step, chunksize):
    """
    This function returns the power spectrum of the desired star.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'flux': Photometry data from the timeserie analysis.
        - 'minfreq': The lower bound for the frequency interval
        - 'maxfreq': The upper bound for the frequency interval
        - 'step': The spacing between frequencies.
        - 'chunksize': Define a chunk for the least mean square Fourier
                       in order to save time.
    """

    # Import modules
    import numpy as np
    from time import time as now

    # Generate cyclic frequencies
    freq = np.arange(minfreq, maxfreq, step)

    # Generate list to store the calculated power
    power = np.zeros((len(freq), 1))

    # Convert frequencies to angular frequencies
    nu = 2 * np.pi * freq

    # Iterate over the frequencies
    timerStart = now()

    # After this many frequencies, print progress info
    print_every = 75e6 // len(time)

    # Ensure chunksize divides print_every
    print_every = (print_every // chunksize) * chunksize

    for i in range(0, len(nu), chunksize):
        # Define chunk
        j = min(i + chunksize, len(nu))
        rows = j - i

        if i % print_every == 0:
            # Info-print
            elapsedTime = now() - timerStart
            if i == 0:
                totalTime = 462 / 6000 * len(nu)
            else:
                totalTime = (elapsedTime / i) * len(nu)

            print("Progress: %.2f%% (%d..%d of %d)  "
                  "Elapsed: %.2f s  Estimated total: %.2f s"
                  % (np.divide(100.0*i, len(nu)), i, j, len(nu),
                     elapsedTime, totalTime))

        """
        The outer product is calculated. This way, the product between
        time and ang. freq. will be calculated elementwise; one column
        per frequency. This is done in order to save computing time.
        """
        nutime = np.outer(time, nu[i:j])

        """
        An array with the measured flux is made so it has the same size
        as "nutime", since we want to multiply the two.
        """
        fluxrep = np.repeat(flux[:, np.newaxis], repeats=rows, axis=1)

        # The Fourier subroutine
        sin_nutime = np.sin(nutime)
        cos_nutime = np.cos(nutime)

        s = np.sum(sin_nutime * fluxrep, axis=0)
        c = np.sum(cos_nutime * fluxrep, axis=0)
        ss = np.sum(sin_nutime ** 2, axis=0)
        cc = np.sum(cos_nutime ** 2, axis=0)
        sc = np.sum(sin_nutime * cos_nutime, axis=0)

        alpha = ((s*cc)-(c*sc))/((ss*cc)-(sc**2))
        beta = ((c*ss)-(s*sc))/((ss*cc)-(sc**2))

        power[i:j] = np.reshape(alpha**2 + beta**2, (rows, 1))
    
    power = power.reshape(-1, 1)
    freq = freq.reshape(-1, 1)
    elapsedTime = now() - timerStart
    print('Computed power spectrum with chunk size %d in %.2f s'
          % (chunksize, elapsedTime))

    return (freq, power)
