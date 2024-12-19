import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from scipy.stats import zscore
import glob
import noisereduce as nr

#Read data
path='../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
files=[s.split('.')[0] for s in os.listdir(path) if '.txt' in s]

audio_file = glob.glob(os.path.join(path, "*.wav"))
y, sr = librosa.load(audio_file[0], sr = 22050)

librosa.display.waveshow(y,sr=sr)
plt.show()

fft_result = np.fft.fft(y)
magnitude = np.abs(fft_result) 
frequencies = np.linspace(0, sr, len(magnitude))

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

n_fft = 2048 
hop_length = 512  
stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label='Amplitude (dB)')
plt.show()

MFFCs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.colorbar()
plt.show()

print()