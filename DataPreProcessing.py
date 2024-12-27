from scipy.signal import butter, sosfilt
import numpy as np
import librosa
import glob
import os
from PyEMD import EMD
import pandas as pd
from scipy.io.wavfile import read

#load audio file
def load_audio_files(path):
    audio_files = glob.glob(os.path.join(path, "*.wav"))
    audio_signals = []

    for file in audio_files:
        try:
            # Load .wav file
            fs, audio = read(file)
            if audio.ndim > 1:  # Handle stereo audio
                audio = audio.mean(axis=1)  # Convert to mono by averaging channels
            audio_signals.append(audio)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return audio_signals

#trim the audio and pad if needed
def trim_and_pad_audio(audio_signals, start_time, end_time, sr):

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    target_length = end_sample - start_sample

    trimmed_audio = []
    for audio in audio_signals:
        # Trim the signal
        trimmed = audio[start_sample:end_sample]

        # Pad if necessary
        if len(trimmed) < target_length:
            trimmed = np.pad(trimmed, (0, target_length - len(trimmed)))

        trimmed_audio.append(trimmed)
    return np.array(trimmed_audio)

#butterworth filtering
def butterworth_filter(audio_batch, lowcut=20, highcut=2000, fs=8000, order=6):

    # Normalize cutoff frequencies
    nyquist = fs * 0.5
    low, high = lowcut / nyquist, highcut / nyquist

    # Design the filter
    sos = butter(order, [low, high], btype='band', output='sos')

    # Apply the filter to each signal in the batch
    return np.array([sosfilt(sos, audio) for audio in audio_batch])

#spectral substraction
def spectral_subtraction(audio, lambda_n=0.8, k=5, frame_size=1024, hop_size=512):

    #Perform STFT 
    D = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)

    # Initialize noise estimation
    noise_spectrum_frames = []
    noise_estimate = np.zeros_like(D) 

    # Estimate the noise spectrum over first k frame
    for i in range(k):
        noise_spectrum_frames.append(D[:, i])  

    # Estimate the noise spectrum from the frames
    noise_spectrum = np.mean(np.abs(noise_spectrum_frames), axis=0)

    # Apply spectral subtraction
    for i in range(k, D.shape[1]):
    # Get current frame noise spectrum
      Nk = D[:, i]

    # Low-pass filter the noise estimate
      if i == k:
        Nk_minus_1 = noise_spectrum
      else:
        Nk_minus_1 = noise_estimate[:, i - 1]

    # Update noise spectrum using weighted average
      noise_spectrum = lambda_n * np.abs(Nk_minus_1) + (1 - lambda_n) * np.abs(Nk)

    # Subtract the noise estimate from the noisy signal 
      D[:, i] = np.maximum(np.abs(D[:, i]) - noise_spectrum, 0)

    cleaned_signal = librosa.istft(D, hop_length=hop_size)
    return cleaned_signal

def denoise_audio(audio, noise_estimate=None):
    audio_denoised = spectral_subtraction(audio)
    return audio_denoised

#emd 
def emd(audio, max_imfs):
    emd = EMD()
    imfs = emd(audio)
    return imfs[:max_imfs]

#rank the imfs
def rank_imfs_by_energy(imfs):
    energy = [np.sum(imf ** 2) for imf in imfs]  
    ranked_indices = np.argsort(energy)[::-1] 
    return ranked_indices

#select imfs
def select_imfs_by_energy(imfs, top_n=3):
    ranked_indices = rank_imfs_by_energy(imfs)
    selected_indices = ranked_indices[:top_n]  # Select top `N` IMFs
    return selected_indices

#reconstruct signal by imfs
def reconstruct_signal(audio, imfs, selected_imfs):
    if len(selected_imfs) == 0:
        print("No IMFs selected. Returning the original audio.")
        return audio

    reconstructed_signal = np.zeros_like(imfs[0])  # Initialize with zeros
    for i in selected_imfs:
        reconstructed_signal += imfs[i]
    return reconstructed_signal

#pad signal after reconstructing
def pad_signal(signal, target_length):
    current_length = len(signal)
    if current_length < target_length:
        padding_length = target_length - current_length
        pad_width = (0, padding_length)  # Pad only at the end
        return np.pad(signal, pad_width, mode='constant', constant_values=0)
    elif current_length > target_length:
        # Trim the signal if it's longer
        return signal[:target_length]
    else:
        # No change
        return signal

#merge audio into csv file
def merge_with_csv(csv_path, audio_files, output_csv_path):
    df = pd.read_csv(csv_path)
    if len(df) != len(audio_files):
        raise ValueError("Mismatch between CSV rows and processed audio files.")
    
    df['denoised_audio'] = [np.array(audio).tolist() for audio in audio_files]

    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

def main():

  path = 'Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files'
  audio_files = glob.glob(os.path.join(path, "*.wav"))
  csv_path = 'df.csv'

  #params
  sr = 8000
  start_time = 3 #limit 6s
  end_time = 9
  max_imfs = 10

  #perform denoising process
  audio = load_audio_files(path)
  trim_audio = trim_and_pad_audio(audio,start_time,end_time,sr)
  audio_batch = np.stack(trim_audio) 
  filtered_batch = butterworth_filter(audio_batch)
  denoised_audio = denoise_audio(filtered_batch)

  processed_signals = []
  for i, audio in enumerate(denoised_audio):
    imfs = emd(audio, max_imfs=10)

    ranked_imfs = rank_imfs_by_energy(imfs)

    selected_imfs = select_imfs_by_energy(imfs, top_n=3)

    reconstructed_audio = reconstruct_signal(audio, imfs, selected_imfs)

    processed_signals.append(reconstructed_audio)

  # Apply padding to all signals
  processed_signals_padded = np.array([pad_signal(signal, sr) for signal in processed_signals])

  new_path = 'df_updated_padded.csv'
  merge_with_csv(csv_path,processed_signals_padded,new_path)

if __name__ == '__main__':
    main()





