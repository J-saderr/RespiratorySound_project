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
from sklearn.model_selection import train_test_split 
import keras
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Resampled class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

def prepare_datasets(X, y, test_size, validation_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    X_train = X_train.reshape(X_train.shape[0], 13, 1, 1)  
    X_validation = X_validation.reshape(X_validation.shape[0], 13, 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 13, 1, 1)
    return X_train, X_validation, X_test, y_train,y_validation, y_test

def build_model(input_shape):
    model = keras.Sequential()
    
    # 1st Conv2D layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd Conv2D layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd Conv2D layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.BatchNormalization())

    #fatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
def predict(model,X,y):
    X = X[np.newaxis,...]
    prediction = model.predict(X)
    predict_index = np.argmax(prediction,axis =1)
    print(f'Expected index: {y}, Predicted index: {predict_index[0]}')
    return

def calculate_metrics(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        score = (sensitivity + specificity) / 2
        return sensitivity, specificity, score, accuracy
    except ValueError:
        print("Error: Metrics calculation failed due to zero samples.")
        return 0, 0, 0, 0

def main():
    path = '../RespiratorySound_project/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
    audio_files = glob.glob(os.path.join(path, "*.wav"))

    df = pd.read_csv('df.csv')
    print(df['Diagnosis'].value_counts())

    features = []
    for audio_file in audio_files:
        mfcc = extract_mfcc(audio_file)
        features.append(mfcc)

    X = np.array(features)
    y = df['Diagnosis'].values

    X, y = balance_data(X, y)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(X, y, 0.3, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    optimizer = keras.optimizers.Adam(learning_rate = 0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    test_error,test_accuracy = model.evaluate(X_test, y_test,verbose=1)
    print(f'Accyracy= {test_accuracy}')
    
    # Normalize test data if necessary
    X_test_normalized = X_test / np.max(X_test)

    # Make prediction
    X_sample = X_test_normalized[100]
    y_sample = y_test[100]
    predict(model, X_sample, y_sample)

    # Evaluate and calculate metrics
    y_pred = (model.predict(X_test_normalized) > 0.5).astype(int)
    sensitivity, specificity, score, accuracy = calculate_metrics(y_test, y_pred)
    print(f"Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%, Score: {score:.2f}%, Accuracy: {accuracy:.2f}%")
    y_pred = (model.predict(X_test_normalized) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == '__main__':
    main()