import numpy as np
import sounddevice as sd
from scipy.io import wavfile

import threading
import time

print(sd.query_devices())
sd.default.device = 24

sample_rate = 44100
chunk_size = 8192

buffer_size = sample_rate * 8  # ~8 seconds of audio
audio_buffer = np.zeros(buffer_size)
write_pos = 0

bpm_history = []

def callback(indata: np.ndarray, frames, time, status):
    global write_pos

    chunk = indata[:, 0]

    n = len(chunk)
    end = write_pos + n

    if end < buffer_size:
        audio_buffer[write_pos:end] = chunk
    else:
        split = buffer_size - write_pos
        audio_buffer[write_pos:] = chunk[:split]
        audio_buffer[:end % buffer_size] = chunk[split:]

    write_pos = end % buffer_size

def onset_envelope(signal, frame_size=1024, hop_size=512):
    env = []

    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i:i+frame_size]
        env.append(np.sum(frame**2))

    env = np.array(env)

    env = np.diff(env)
    env = np.maximum(env, 0)

    return env

def estimate_bpm_from_input(data, sr):
    #print("Level:", np.mean(np.abs(data)))

    if np.mean(np.abs(data)) < 0.0001:
        return None
    
    hop = 512

    env = onset_envelope(data, hop_size=hop)

    # smooth a bit
    env = np.convolve(env, np.ones(4)/4, mode='same')

    autocorr = np.correlate(env, env, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    min_bpm, max_bpm = 60, 180

    min_lag = int((60 / max_bpm) * sr / hop)
    max_lag = int((60 / min_bpm) * sr / hop)

    autocorr[:min_lag] = 0
    autocorr[max_lag:] = 0

    lag = np.argmax(autocorr)
    bpm = 60 * sr / (lag * hop)

    return bpm

def bpm_thread():
    while True:
        time.sleep(1.0)

        data = audio_buffer.copy()

        instant_bpm = estimate_bpm_from_input(data, sample_rate)
        if(instant_bpm != None):
            bpm_history.append(instant_bpm)

        if(len(bpm_history) > 5):
            bpm_history.pop(0)

        bpm = np.median(bpm_history)
        print(f"BPM: {bpm:.1f}")

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=callback,
    blocksize=chunk_size,
    dtype='float32'
)

with stream:
    bpm = threading.Thread(target=bpm_thread)
    bpm.start()
    try:
        while True:
            sd.sleep(1000)

    except KeyboardInterrupt:
        bpm.join()
        pass