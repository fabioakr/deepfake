'''Name, explanation, etc

Passo1: criar o venv
Passo2: instalar as dependências
pip3 install librosa ipython

'''

import librosa
import IPython.display as ipd
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13, sr=16000):
    """
    Extracts MFCC features from an audio file.

    Parameters:
    - audio_path: str, path to the audio file.
    - n_mfcc: int, number of MFCC features to extract.
    - sr: int, sample rate for loading the audio file.

    Returns:
    - mfccs: numpy array of shape (n_mfcc, T) where T is the number of frames.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

#Main function to demonstrate usage
def main():
    print("Extracting MFCC features from audio file...")

if __name__ == "__main__":
    main()
