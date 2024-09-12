import madmom
from madmom.audio.signal import Signal
from madmom.audio.filters import LogarithmicFilterbank
import numpy as np
import matplotlib.pyplot as plt

def load_audio(audio_file, sample_rate=44100, num_channels=1):
    """
    Load an audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        sample_rate (int): Sample rate of the audio file.
        num_channels (int): Number of audio channels.
    
    Returns:
        Signal: Loaded audio signal.
    """
    return Signal(audio_file, sample_rate=sample_rate, num_channels=num_channels)

def apply_filterbank(signal, num_bands=24, fmin=30, fmax=17000):
    """
    Apply a logarithmic filterbank to the audio signal.
    
    Args:
        signal (Signal): Audio signal.
        num_bands (int): Number of frequency bands.
        fmin (int): Minimum frequency for the filterbank.
        fmax (int): Maximum frequency for the filterbank.
    
    Returns:
        np.ndarray: Filterbank processed signal.
    """
    filterbank = LogarithmicFilterbank(num_bands=num_bands, fmin=fmin, fmax=fmax)
    stft = madmom.audio.stft.STFT(signal)
    return filterbank.process(stft)

def plot_filtered_signal(filtered_signal):
    """
    Plot the preprocessed signal.
    
    Args:
        filtered_signal (np.ndarray): Filterbank processed signal.
    """
    plt.imshow(np.log(1 + np.abs(filtered_signal.T)), aspect='auto', origin='lower')
    plt.title('Logarithmic Filterbank Output')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency Bands')
    plt.colorbar(label='Magnitude (Log scale)')
    plt.show()

# Example usage
if __name__ == "__main__":
    audio_file = 'your_audio_file.wav'
    
    # Load and preprocess the audio
    signal = load_audio(audio_file)
    filtered_signal = apply_filterbank(signal)
    
    # Plot the preprocessed signal
    plot_filtered_signal(filtered_signal)
