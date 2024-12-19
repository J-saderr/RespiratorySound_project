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
df_no_diagnosis = pd.read_csv('../RespiratorySound_proj/demographic_info.txt', 
                               names = ['Patient id', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'], delimiter = ' ')

diagnosis = pd.read_csv('../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', 
                        names = ['Patient id', 'Diagnosis'])

path='../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
files=[s.split('.')[0] for s in os.listdir(path) if '.wav' in s]

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(
        data=[tokens], 
        columns=['Patient id', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment']
    )
    return recording_info

i_list = []
for s in files:
    i = Extract_Annotation_Data(s, path)
    i_list.append(i)

recording_info = pd.concat(i_list, axis=0)
recording_info['Patient id'] = recording_info['Patient id'].apply(pd.to_numeric, errors='coerce')

df = diagnosis.join(recording_info.set_index('Patient id'), on='Patient id', how='left')

audio_files = glob.glob(os.path.join(path, "*.wav"))
for audio_file in audio_files:
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=22050)

    # Create a figure with 4 subplots (one for each feature)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Features of {os.path.basename(audio_file)}")

    # Plot the waveform in the first subplot
    axs[0, 0].set_title("Waveform")
    librosa.display.waveshow(y, sr=sr, ax=axs[0, 0])
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")

    # FFT (Frequency Spectrum) in the second subplot
    fft_result = np.fft.fft(y)
    magnitude = np.abs(fft_result)  # Magnitude of FFT
    frequencies = np.linspace(0, sr, len(magnitude))
    axs[0, 1].set_title("Frequency Spectrum")
    axs[0, 1].plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])  # Only plot positive frequencies
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Magnitude")

    # STFT (Spectrogram) in the third subplot
    n_fft = 2048
    hop_length = 512
    stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    axs[1, 0].set_title("Log Spectrogram")
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, ax=axs[1, 0])
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].colorbar(label='Amplitude (dB)')

    # MFCC in the fourth subplot
    MFFCs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    axs[1, 1].set_title("MFCC")
    librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length, ax=axs[1, 1])
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("MFCC Coefficients")
    axs[1, 1].colorbar()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the title space
    plt.show()



'''
adults = df_no_diagnosis[df_no_diagnosis['Age'] >= 18].drop(['Child Weight (kg)', 'Child Height (cm)'], axis=1)
children = df_no_diagnosis[df_no_diagnosis['Age'] < 18].drop(['Adult BMI (kg/m2)'], axis=1)
adults = adults.join(df.set_index('Patient id'), on='Patient id', how='left')
children = children.join(df.set_index('Patient id'), on='Patient id', how='left')

print("\nShow first 5 rows for adults dataset:")
print(adults.head())
print("\nShow first 5 rows for children dataset:")
print(children.head())

print("\nData types for adults dataset:")
print(adults.info())
print("\nData types for children dataset:")
print(children.info())

print("\nCheck missing values for adults dataset:")
print(adults.isnull().sum())
print("\nCheck missing values for children dataset:")
print(children.isnull().sum())

#Add values to missing values
children = children.fillna(children.mean(numeric_only=True))
adults = adults.fillna(adults.mean(numeric_only=True))

print("\nCheck duplicated values for adults dataset:")
print(adults[adults.duplicated(keep=False)].sum())
print("\nCheck duplicated values for children dataset:")
print(children[children.duplicated(keep=False)].sum())

print("\nDescibe the data:")
print(adults.describe().T)
print(children.describe().T)

print("\nCheck outliers")


#Boxplot before cleaned
continuous = ['Patient id', 'Age', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)']
fig, axes = plt.subplots(len(continuous), 1, figsize=(10, len(continuous) * 2))
for i, ax in zip(continuous, axes):
    sns.boxplot(x=df1[i], color='#A4161A', linewidth=1, ax=ax)
    ax.set_title(f'Boxplot of {i}')
    ax.set_xlabel(i)
    ax.set_ylabel('')
plt.tight_layout()
#plt.show()

#Z-score to delete outliers
Z_THRESHOLD = 3
for col in continuous:
    if df1[col].dtype in ['float64', 'int64']:
        df1_clean = df1[(abs(zscore(df1[col].dropna())) <= Z_THRESHOLD)]

print("Data after removing outliers:")
print(df1_clean.describe().T)

#Boxplot after cleaned
fig, axes = plt.subplots(len(continuous), 1, figsize=(10, len(continuous) * 2))

for i, ax in zip(continuous, axes):
    sns.boxplot(x=df1_clean[i], color='#A4161A', linewidth=1, ax=ax)
    ax.set_title(f'Boxplot of {i} (Outliers removed)')
    ax.set_xlabel(i)
    ax.set_ylabel('')
    
plt.tight_layout()
#plt.show()

print()
'''
