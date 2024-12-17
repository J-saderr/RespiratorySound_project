import numpy
import pandas as pd
df_no_diagnosis = pd.read_csv('/Users/vothao/Documents/Đại học/Năm 3/Học kì 1/AI/Bản sao RespiratorySound/demographic_info.txt', 
                               names = ['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'], delimiter = ' ')
diagnosis = pd.read_csv('/Users/vothao/Documents/Đại học/Năm 3/Học kì 1/AI/Bản sao RespiratorySound/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', 
                        names = ['Patient number', 'Diagnosis'])

# Check if the files are loaded correctly
print("Demographic Data (First 5 rows):")
print(df_no_diagnosis.head())  # Display the first 5 rows of demographic data
print("\nDiagnosis Data (First 5 rows):")
print(diagnosis.head())  # Display the first 5 rows of diagnosis data

# Join the DataFrames
df = df_no_diagnosis.join(diagnosis.set_index('Patient number'), on='Patient number', how='left')

# Check the first few rows of the combined DataFrame
print("\nCombined Data (First 5 rows):")
print(df.head())

# Print the value counts of the 'Diagnosis' column
print("\nDiagnosis Value Counts:")
print(df['Diagnosis'].value_counts())