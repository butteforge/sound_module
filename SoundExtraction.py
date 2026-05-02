#
# This file extracts sound data from a file and creates some specific graphs
#   first and second graph are channel graphs that display amplitude of sound over time
#   third graph is the amplitude over frequency using FFT to transform the time domain to freq domain
#   fourth graph is a spectogram showing frequency levels over time
#

import numpy

from scipy.io import wavfile
import matplotlib.pyplot as plt

rate, data = wavfile.read("win7.wav")

print(f"sampling rate: {rate}")


#channels = data.shape[1]
#print(f"channels: {channels}")

file_length_seconds = data.shape[0] / rate

print(f"seconds: {file_length_seconds}")

plot_time = False

spec = True


fig, axs = plt.subplots(2, 2)




time = numpy.linspace(0, len(data) / rate, num=len(data))

x = numpy.arange(-1, 5)
y = numpy.array(data)

axs[0, 0].plot(time, data[:])
axs[0, 0].set_xlabel("Time (seconds)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title("Audio Waveform (Channel 0)")

axs[0, 1].plot(time, data[:])
axs[0, 1].set_xlabel("Time (seconds)")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 1].set_title("Audio Waveform (Channel 1)")

if(len(data.shape) > 1):
    data = data[:, 0] # only take the first channel of data, convert it from 2d to 1d

fft = numpy.fft.fft(data)
fft = numpy.abs(fft) / len(data) # normalize data 
freqs = numpy.fft.fftfreq(len(fft), 1/rate)

half = len(freqs) // 2


axs[1,0].specgram(data, Fs=rate)

axs[1,1].plot(freqs[:half], numpy.abs(fft[:half]))
axs[1,1].set_xlabel("Frequency (Hz)")
axs[1,1].set_ylabel("Magnitude")
axs[1,1].set_title("Frequency Spectrum")

plt.show()