import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import read, write
import sounddevice as sd

def load_audio(filename):
    sampling_rate, audio = read(filename)
    return audio, sampling_rate

def play_audio(audio, sampling_rate):
    sd.play(audio, sampling_rate)
    sd.wait()

def shift_signal_in_frequency_domain(audio, sampling_rate, frequency_shift):
    # Compute the Fourier Transform of the audio signal
    N = len(audio)
    audio_ft = fft(audio)
    
    # Compute the frequency spectrum
    freqs = fftfreq(N, 1 / sampling_rate)
    
    # Create the complex exponential for shifting
    shift = np.exp(1j * 2 * np.pi * frequency_shift * np.arange(N) / N)
    
    # Apply the shift
    shifted_audio_ft = audio_ft * shift
    
    # Compute the inverse Fourier Transform to get the time-domain signal
    shifted_audio = ifft(shifted_audio_ft)
    
    return np.real(shifted_audio)

# Example usage
filename = 'recording.wav'
audio, sampling_rate = load_audio(filename)

# Shift the signal in the frequency domain by 1000 Hz
frequency_shift = 1000
shifted_audio = shift_signal_in_frequency_domain(audio, sampling_rate, frequency_shift)

# Save the shifted signal
write('shifted_recording.wav', sampling_rate, shifted_audio.astype(np.int16))

# Play the shifted signal
play_audio(shifted_audio, sampling_rate)

# Plot the original and shifted signals for comparison
def plot_signals(original, shifted, sampling_rate):
    t = np.linspace(0, len(original) / sampling_rate, num=len(original))
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t, original)
    plt.title("Original Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, shifted)
    plt.title("Shifted Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    
    plt.tight_layout()
    plt.show()

plot_signals(audio, shifted_audio, sampling_rate)
