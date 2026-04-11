import sounddevice as sd
import numpy as np

from scipy.io import wavfile

print(sd.query_devices())

sd.default.device = 24

sample_rate = 44100
chunk_size = 1024

buffer = []

def callback(indata, frames, time, status):
    # indata is a NumPy array (frames × channels)
    audio_chunk = indata[:, 0].copy()  # mono
    buffer.append(audio_chunk)

    # example: print loudness
    volume = np.linalg.norm(audio_chunk)
    #print(f"Volume: {volume:.3f}")

    print(f"Freq: {freq_to_note(get_dominant_freq(audio_chunk, sample_rate))}")


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

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=callback,
    blocksize=chunk_size,
    dtype='float32'
)

with stream:
    print("Recording... Press Ctrl+C to stop")
    try:
        while True:
            sd.sleep(1000)

    except KeyboardInterrupt:
        audio = np.concatenate(buffer)

        # ensure correct range
        audio = np.clip(audio, -1.0, 1.0)

        #   convert to int16
        audio_float32 = (audio * 32767).astype(np.float32)

        wavfile.write("mic_recording3.wav", sample_rate, audio_float32)