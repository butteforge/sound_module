#
# This file extracts sound data from a file and reads the most prominant frequency at a certai interval
#

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

rate, data = wavfile.read("mic_recording3.wav")

if (len(data.shape) > 1):
    data = data[:,0]

chunk_size = int(44100/60) # number of samples to find dominant freq

def get_dominant_freq(chunk, rate):
    fft = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(fft), 1 / rate)

    magnitude = np.abs(fft)

    # Only positive frequencies
    half = len(freqs) // 2
    peak_index = np.argmax(magnitude[:half])

    return freqs[peak_index]

def freq_to_note(freq):
    if freq <= 0:
        return None

    # Reference: A4 = 440 Hz
    A4 = 440.0

    # Calculate number of semitones from A4
    n = 12 * np.log2(freq / A4)

    # Round to nearest semitone
    n = int(round(n))

    # Note names
    notes = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]

    # Calculate note index
    note_index = (n + 9) % 12  # A is index 9

    # Calculate octave
    octave = 4 + ((n + 9) // 12)

    return f"{notes[note_index]}{octave}"

sd.play(data, rate)
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]

    if len(chunk) < chunk_size:
        break

    freq = get_dominant_freq(chunk, rate)
    print(f"Dominant frequency: {freq_to_note(freq)}, Time interval {i / rate}-{(i+chunk_size)/rate}")
    
    
sd.wait()
    
