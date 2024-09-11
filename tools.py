import librosa

def load_audio_file(file_path):
    '''
    Load an audio file and return the audio data and the sample rate
    '''
    return librosa.load(file_path, sr=None)

def convert_to_spectrogram(audio_data, sample_rate):
    '''
    Convert audio data to a spectrogram
    '''
    return librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)

if __name__ == '__main__':
    pass