import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from scipy.stats import zscore

#Read data
df_no_diagnosis = pd.read_csv('../RespiratorySound_proj/demographic_info.txt', 
                               names = ['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'], delimiter = ' ')
diagnosis = pd.read_csv('../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', 
                        names = ['Patient number', 'Diagnosis'])
df = df_no_diagnosis.join(diagnosis.set_index('Patient number'), on='Patient number', how='left')

path='../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
files=[s.split('.')[0] for s in os.listdir(path) if '.txt' in s]

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)

i_list = []
rec_annotations = []
rec_annotations_dict = {}
for s in files:
    (i,a) = Extract_Annotation_Data(s, path)
    i_list.append(i)
    rec_annotations.append(a)
    rec_annotations_dict[s] = a
recording_info = pd.concat(i_list, axis = 0)
print(recording_info.tail())
print(recording_info.info())
recording_info = recording_info.apply(pd.to_numeric, errors='coerce')
'''
print("Display diagnosis data first 5 rows:")
print(df_no_diagnosis.head())  
print("\nDisplay diagnosis data first 5 rows:")
print(diagnosis.head()) 

print("\nCombined data first 5 rows:")
print(df.head())

print("\nDiagnosis value counts:")
print(df['Diagnosis'].value_counts())

print("\nCheck missing values:")
print(df.isnull().sum())

#Add values to missing
df = df.fillna(df.mean(numeric_only=True))
df['Sex'] = df['Sex'].fillna('Unknown')

print("\nCheck duplicated values:")
print(df[df.duplicated(keep=False)].sum())

print("\nInfomation of data:")
print(df.info())

print("\nDescibe the data:")
print(df.describe().T)

print("\nCheck outliers")
#Boxplot before cleaned
continuous = ['Patient number', 'Age', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)']
fig, axes = plt.subplots(len(continuous), 1, figsize=(10, len(continuous) * 2))
for i, ax in zip(continuous, axes):
    sns.boxplot(x=df[i], color='#A4161A', linewidth=1, ax=ax)
    ax.set_title(f'Boxplot of {i}')
    ax.set_xlabel(i)
    ax.set_ylabel('')
plt.tight_layout()
#plt.show()

#Z-score to delete outliers
Z_THRESHOLD = 3
for col in continuous:
    if df[col].dtype in ['float64', 'int64']:
        df_clean = df[(abs(zscore(df[col].dropna())) <= Z_THRESHOLD)]

print("Data after removing outliers:")
print(df_clean.describe().T)

#Boxplot after cleaned
fig, axes = plt.subplots(len(continuous), 1, figsize=(10, len(continuous) * 2))

for i, ax in zip(continuous, axes):
    sns.boxplot(x=df_clean[i], color='#A4161A', linewidth=1, ax=ax)
    ax.set_title(f'Boxplot of {i} (Outliers removed)')
    ax.set_xlabel(i)
    ax.set_ylabel('')
    
plt.tight_layout()
#plt.show()

print()
'''
