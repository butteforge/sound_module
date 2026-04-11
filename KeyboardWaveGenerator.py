import numpy as np
import sounddevice as sd
from pynput import keyboard

sample_rate = 44100
current_freq = 440.0  # default A4
phase = 0.0
wave = np.sin(2 * np.pi * 0.0 * 1.0)

# Map keyboard keys to frequencies
key_map = {
    'a': 261.63,  # C4
    's': 293.66,  # D4
    'd': 329.63,  # E4
    'f': 349.23,  # F4
    'g': 392.00,  # G4
    'h': 440.00,  # A4
    'j': 493.88,  # B4
    'k': 523.25,  # C5
    ' ': -1,
}

def audio_callback(outdata, frames, time, status):
    global phase, current_freq, wave

    t = (np.arange(frames) + phase) / sample_rate

    wave = np.sin(2 * np.pi * current_freq * t)
    wave /= np.max(np.abs(wave))

    outdata[:] = wave.reshape(-1, 1)
    phase += frames
    phase %= sample_rate  # prevent overflow


def on_press(key):
    global current_freq

    try:
        if key.char in key_map:
            current_freq = key_map[key.char]
            print(f"Playing {current_freq:.2f} Hz")
    except AttributeError:
        pass


def on_release(key):
    global current_freq

    # Stop sound when key released
    try:
        if key.char in key_map:
            current_freq = 0.0
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        return False  # stop listener


# Start audio stream
stream = sd.OutputStream(
    channels=1,
    callback=audio_callback,
    samplerate=sample_rate
)

with stream:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()