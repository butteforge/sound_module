#
# This file outputs a singular note
#

import numpy as np
import sounddevice as sd

from scipy.io import wavfile

# Parameters
sample_rate = 44100  # samples per second
duration = 2.0       # seconds
frequency = 493.883    # Hz (B4 note)

# Time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#data array

# Generate sine wave
wave = np.sin(2 * np.pi * frequency * t)

# Play sound
sd.play(wave, sample_rate)
sd.wait()