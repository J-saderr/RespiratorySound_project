import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from scipy.stats import zscore
import glob
import noisereduce as nr
from concurrent.futures import ProcessPoolExecutor

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(
        data=[tokens], 
        columns=['Patient id', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment']
    )
    return recording_info

def process_audio_file(file):
    y, sr = librosa.load(file, sr=22050)
    fft_result = np.fft.fft(y)
    magnitude = np.abs(fft_result)
    frequencies = np.linspace(0, sr, len(magnitude))
    patient_id = int(os.path.basename(file).split('_')[0])
    return {
        'Patient id': patient_id,
        'Frequencies': frequencies.tolist() 
    }

# Function to plot waveform
def plot_waveform(y, sr, title, ax):
    ax.plot(y)
    ax.set_title(f"Waveform - {title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

# Function to plot FFT
def plot_fft(y, sr, title, ax):
    fft_result = np.fft.fft(y)
    magnitude = np.abs(fft_result)
    frequencies = np.linspace(0, sr, len(magnitude))
    ax.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    ax.set_title(f"FFT - {title}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

# Function to plot STFT Spectrogram
def plot_stft(y, sr, title, ax):
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=ax)
    ax.set_title(f"STFT Spectrogram - {title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

# Function to plot MFCC
def plot_mfcc(y, sr, title, ax):
    n_fft = 2048
    hop_length = 512
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis="time", ax=ax)
    ax.set_title(f"MFCC - {title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficients")

# Function to plot all audio analysis
def plot_audio_analysis(y, sr, title, ax):
    plot_waveform(y, sr, title, ax[0])
    plot_fft(y, sr, title, ax[1])
    plot_stft(y, sr, title, ax[2])
    plot_mfcc(y, sr, title, ax[3])

def main():
    # Read data
    diagnosis = pd.read_csv('../RespiratorySound_project/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', 
                            names = ['Patient id', 'Diagnosis'])
    path = '../RespiratorySound_project/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
    files = [s.split('.')[0] for s in os.listdir(path) if '.txt' in s]

    # Extract recording information
    i_list = [Extract_Annotation_Data(s, path) for s in files]
    recording_info = pd.concat(i_list, axis=0)
    recording_info['Patient id'] = recording_info['Patient id'].apply(pd.to_numeric, errors='coerce')
    recording_info = recording_info.drop(['Recording index', 'Recording equipment', 'Acquisition mode'], axis=1, errors='ignore')

    # Process audio files in parallel
    audio_files = glob.glob(os.path.join(path, "*.wav"))
    #with ProcessPoolExecutor() as executor:
    #    frequency_data = list(executor.map(process_audio_file, audio_files))

    # Create frequency DataFrame
    #frequency_df = pd.DataFrame(frequency_data)

    # Merge the data frames
    df = diagnosis.join(recording_info.set_index('Patient id'), on='Patient id', how='left')
    #df = df.merge(frequency_df, on='Patient id', how='left')

    # Separate Not COPD and COPD files
    audio_files_copd = [audio_files[3]]

    fig, axes = plt.subplots(len(audio_files_copd), 4, figsize=(20, 5 * len(audio_files_copd))) 

    for i, file in enumerate(audio_files_copd):
        y, sr = librosa.load(file, sr=22050)
        patient_id = int(os.path.basename(file).split('_')[0])
        title = 'COPD'
        plot_audio_analysis(y, sr, title, axes)

    plt.tight_layout()
    plt.show()

    audio_files_not_copd = [
        (audio_files[0], 'URTI'),
        (audio_files[1], 'Healthy'),
        (audio_files[2], 'Asthma'),
        (audio_files[7], 'LRTI'),
        (audio_files[10], 'Bronchiectasis'),
        (audio_files[21], 'Pneumonia'),
        (audio_files[48], 'Bronchiolitis') 
    ]

    # Plotting for Non-COPD Cases
    fig, axes = plt.subplots(len(audio_files_not_copd), 4, figsize=(24, 3 * len(audio_files_not_copd)))

    # Loop through non-COPD cases and plot
    for i, (file, diagnosis) in enumerate(audio_files_not_copd):
        y, sr = librosa.load(file, sr=22050)
        title = f'{diagnosis}' 
        plot_audio_analysis(y, sr, title, axes[i]) 

    plt.tight_layout()
    plt.show()

    # Replace 'COPD' with 1, and others with 0
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 1 if x == 'COPD' else 0)
    df.to_csv('df.csv', index=False)

if __name__ == '__main__':
    main()