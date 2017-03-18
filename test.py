import numpy as np
import powerspectrum as psp
import ts_powerspectrum as ts
from matplotlib import pyplot as plt

time = np.arange(0, 20, 0.01)
flux = 4 * np.sin(12*np.pi*time) + 2 * np.sin(2 * np.pi * 7 * time)

minfreq = 1
maxfreq = 200
step = 1 / (4 * (time[-1] - time[0]))
chunksize = 500

freq, power = psp.power(time, flux, minfreq, maxfreq, step, chunksize)
freqts, powerts, alpha, beta = ts.power_spectrum(
    time, flux, minfreq=minfreq, maxfreq=maxfreq)

assert freq.shape == freqts.shape
print(np.allclose(freq, freqts))
print(np.allclose(power, powerts))

plt.plot(freq, power)
plt.show()
