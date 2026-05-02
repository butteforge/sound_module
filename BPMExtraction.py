
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

rate, data = wavfile.read("crab-wav.wav")

# convert data to mono
data = data.mean(axis=1) if data.ndim > 1 else data

def onset_envelope(signal, sr, frame_size=1024, hop_size=512):
    envelope = []

    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i:i+frame_size]
        energy = np.sum(frame**2)
        envelope.append(energy)

    envelope = np.array(envelope)

    envelope=np.diff(envelope)
    envelope=np.maximum(envelope, 0)

    return envelope

def estimate_bpm(onset_env, sr, hop_size):
    autocorr = np.correlate(onset_env, onset_env, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    lags = np.arange(len(autocorr))

    min_bpm = 60
    max_bpm = 200

    min_lag = int((60/max_bpm) * sr / hop_size)
    max_lag = int((60/min_bpm) * sr / hop_size)

    autocorr[:min_lag] = 0
    autocorr[max_lag:] = 0

    best_lag = np.argmax(autocorr)

    bpm = 60 * sr / (best_lag * hop_size)
    return bpm


sr = rate
hop_size = 512

env = onset_envelope(data, sr, hop_size=hop_size)
bpm = estimate_bpm(env, sr, hop_size)

print(f"Estimated BPM: {bpm:.2f}")