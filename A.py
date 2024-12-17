import numpy
import pandas as pd
df_no_diagnosis = pd.read_csv('../RespiratorySound_proj/demographic_info.txt', 
                               names = ['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'], delimiter = ' ')
diagnosis = pd.read_csv('../RespiratorySound_proj/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', 
                        names = ['Patient number', 'Diagnosis'])

print("Display diagnosis data first 5 rows:")
print(df_no_diagnosis.head())  
print("\nDisplay diagnosis data first 5 rows:")
print(diagnosis.head()) 

df = df_no_diagnosis.join(diagnosis.set_index('Patient number'), on='Patient number', how='left')

print("\nCombined data first 5 rows:")
print(df.head())

print("\nDiagnosis value counts:")
print(df['Diagnosis'].value_counts())