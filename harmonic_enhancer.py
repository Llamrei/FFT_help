# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:50:12 2019

Assignment 1. Digital Signal Processing: Fourier Transform 

@authors: 2253390J and 2254735R
"""
if __name__ == "__main__":
    # Q1 & 2
    # importing scipy to read and write our wave files
    import scipy.io.wavfile as wavfile

    # importing numpy module as np
    import numpy as np

    # importing pyplot module from matplot library to plot variables
    import matplotlib.pyplot as plt

    # fs is the sampling frequency 44.1kHz
    # data holds values from our wav file

    # close previous plots
    plt.close("all")
    fs, data = wavfile.read("original.wav")

    # normalising our data by dividing by highest signed int16 value
    data = data / 32768

    # t= np.linspace(0, 2 * np.pi , len(data))
    # data = data +  np.sin(10000*t)

    # time creates an array of values to correct the x axis for time in seconds
    time = np.linspace(0, len(data) / fs, num=len(data))

    # plotting audio signal in time domain with linear axes
    plt.figure(1, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(time, data)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.title("Time domain plot")
    plt.grid()
    plt.show()

    # Fast fourier transform
    # xf is amplitude values for our signal in frequency domain
    # two versions of xf, so the mirror can be removed later in xf_no_mirror
    # xf contains all amplitude values
    # xf_no_mirror will have mirror values removed from frequency domain
    xf_no_mirror = 2 * np.fft.fft(data) / len(data)
    xf = 2 * np.fft.fft(data) / len(data)

    # Removes the mirror in frequency domain by only selecting first half of values
    # mirror occurs at half the number of samples
    xf_no_mirror = xf_no_mirror[0 : int(len(xf_no_mirror) / 2)]

    # Plot with the mirror removed in frequency domain
    # faxis corrects x axis for frequency domain plot
    faxis = np.linspace(0, 44100, len(xf))
    faxis_no_mirror = np.linspace(0, 44100 / 2, int(len(xf) / 2))
    plt.figure(2, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(faxis_no_mirror, abs(xf_no_mirror))
    plt.ylabel("Normalised Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.title("Frequency plot mirror removed")
    plt.grid()
    plt.show()

    # converting amplitude values into decibels for log frequency plot
    db_log_xf_no_mirror = 20 * np.log10(abs(xf_no_mirror))

    # Normalised amplitude converted to dB and plotted against log frequency
    plt.figure(3, figsize=(14, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(faxis_no_mirror, db_log_xf_no_mirror)
    plt.ylabel("dB")
    plt.xlabel("Log Frequency (Hz) ")
    plt.title("Plot of dB vs  log frequency")
    plt.grid(which="minor")
    plt.xscale("log")
    plt.show()

    # removing amplitude values of frequencies 0 to 60Hz.
    # these releate to excess bass in the signal due to the proximity effect
    removeDC1 = int((len(xf) / 44100) * 0)
    removeDC2 = int((len(xf) / 44100) * 60)
    # 60Hz frequency also removed from mirrored side
    xf[removeDC1 : removeDC2 + 1] = 0
    xf[len(xf) - removeDC2 : len(xf) - removeDC1 + 1] = 0
    xf_no_mirror[removeDC1 : removeDC2 + 1] = 0

    # removed frequencies above 10000Hz, values above harmonics
    removeSignals2 = int((len(xf) / 44100) * 10000)
    # 10000Hz frequency also removed mirrored side
    xf[removeSignals2 : int((len(xf) / 2) + 1)] = 0
    xf[int((len(xf) / 2)) : len(xf) - removeSignals2 + 1] = 0
    xf_no_mirror[removeSignals2 : (int(len(xf) / 2)) + 1] = 0

    # creating new variable k to store signal after harmonics are amplified
    k = xf

    # q3
    # variables to find position of fundamental & harmonic frequencies
    # obtain index values of our signal after FFT to find position of frequencies to manipulate
    # 1st fundemental in range 70-120Hz
    f1start = int((len(xf) / 44100) * 70)
    f1end = int((len(xf) / 44100) * 120)

    # 2nd fundamental in range 150 - 210Hz
    f2start = int((len(xf) / 44100) * 150)
    f2end = int((len(xf) / 44100) * 210)

    # 3rd fundamental in range 240 - 290Hz
    f3start = int((len(xf) / 44100) * 240)
    f3end = int((len(xf) / 44100) * 290)

    # 4th fundamental in range 300 - 670Hz
    f4start = int((len(xf) / 44100) * 300)
    f4end = int((len(xf) / 44100) * 670)

    # higher order harmonics are found in frequencies above 1000Hz
    # harmonics found in range 1050 - 1100 Hz
    h1start = int((len(xf) / 44100) * 1050)
    h1end = int((len(xf) / 44100) * 1100)

    # harmonics found in range 1300-1720hz
    h2start = int((len(xf) / 44100) * 1300)
    h2end = int((len(xf) / 44100) * 1720)

    # harmonics found in range 2000 - 2570Hz
    h3start = int((len(xf) / 44100) * 2000)
    h3end = int((len(xf) / 44100) * 2570)

    # harmonics found in range 3000 - 3800Hz
    h4start = int((len(xf) / 44100) * 3000)
    h4end = int((len(xf) / 44100) * 3800)

    # harmonics found in range 4000 - 5100Hz
    h5start = int((len(xf) / 44100) * 4000)
    h5end = int((len(xf) / 44100) * 5100)

    # harmonics found in range 5200 - 5800Hz
    h6start = int((len(xf) / 44100) * 5200)
    h6end = int((len(xf) / 44100) * 5800)

    # increasing amplitude of harmonic frequency ranges

    # increasing fundemental 1050 - 1100Hz
    IndicesToAmplify1 = range(h1start, h1end)
    for n in IndicesToAmplify1:
        k[n] = k[n] * 2

    # increasing fundemental 1050 - 1000Hz on mirror
    IndicesToAmplify15 = range((len(xf) - h1end), (len(xf) - h1start + 1))
    for n in IndicesToAmplify15:
        k[n] = k[n] * 2

    # increasing fundamental 1300 - 1720Hz
    IndicesToAmplify2 = range(h2start, h2end)
    for n in IndicesToAmplify2:
        k[n] = k[n] * 1
    # increasing fundamental 1300 - 1720Hz  on mirror
    IndicesToAmplify25 = range((len(xf) - h2end), (len(xf) - h2start + 1))
    for n in IndicesToAmplify25:
        k[n] = k[n] * 1

    # increasing harmonic 2000 - 2570Hz
    IndicesToAmplify3 = range(h3start, h3end)
    for n in IndicesToAmplify3:
        k[n] = k[n] * 3

    # increasing harmonic 2000 - 2570Hz on mirror
    IndicesToAmplify35 = range((len(xf) - h3end), (len(xf) - h3start + 1))
    for n in IndicesToAmplify35:
        k[n] = k[n] * 3

    # increasing harmonic 3000 - 3800Hz
    IndicesToAmplify4 = range(h4start, h4end)
    for n in IndicesToAmplify4:
        k[n] = k[n] * 3

    # increasing harmonic 3000 - 3800Hz on mirror
    IndicesToAmplify45 = range((len(xf) - h4end), (len(xf) - h4start + 1))
    for n in IndicesToAmplify45:
        k[n] = k[n] * 3

    # increasing harmonic 4000 - 5100Hz
    IndicesToAmplify5 = range(h5start, h5end)
    for n in IndicesToAmplify5:
        k[n] = k[n] * 2
    # increasing harmonic 4000 - 5100Hz on mirror
    IndicesToAmplify55 = range((len(xf) - h5end), (len(xf) - h5start + 1))
    for n in IndicesToAmplify55:
        k[n] = k[n] * 2

    # increasing harmonic 5200 - 5800Hz
    IndicesToAmplify6 = range(h6start, h6end)
    for n in IndicesToAmplify6:
        k[n] = k[n] * 1

    # increasing harmonic 5200 - 5800Hz on mirror
    IndicesToAmplify65 = range((len(xf) - h6end), (len(xf) - h6start + 1))
    for n in IndicesToAmplify65:
        k[n] = k[n] * 1

    # Inverse fourier transform of signal after amplifying frequencies
    # Signal back to time domain
    original_clean = np.fft.ifft(k)
    original_clean = np.real(original_clean)
    # plotting manipulated signal in time domain
    plt.figure(4)
    plt.plot(time, original_clean)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.title("Time domain plot of manipulated signal")
    plt.grid()
    plt.show()

    # importing write function to output wav file from scipy
    from scipy.io.wavfile import write

    # scaling the output to int16
    scaled = np.int16(original_clean / np.max(np.abs(original_clean)) * 32768)
    # writing new wav file
    write("improved.wav", 44100, scaled)

