import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import keras
from sklearn.metrics import confusion_matrix,f1_score, roc_curve, auc
from keras_tuner import RandomSearch

#Mcff
def extract_mfcc(denoised_audio, sr):
    mfcc = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Prepare datasets
def prepare_datasets(X, y, test_size, validation_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    X_train = np.array(X_train)[..., np.newaxis]
    X_validation = np.array(X_validation)[..., np.newaxis]
    X_test = np.array(X_test)[..., np.newaxis]
    return X_train, X_validation, X_test, y_train,y_validation, y_test

# Build model CNN
def build_model(hp):
    model = keras.Sequential()
    # 1st Conv2D layer
    model.add(keras.layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        input_shape=(None, None, 1)
    ))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd Conv2D layer
    model.add(keras.layers.Conv2D(
        filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd Conv2D layer
    model.add(keras.layers.Conv2D(
        filters=hp.Int('filters_3', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.BatchNormalization())

    # Flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('units_1', min_value=64, max_value=256, step=64),
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(
        units=hp.Int('units_2', min_value=32, max_value=128, step=32),
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dropout(0.5))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Prediction
def predict(model,X,y):
    X = X[np.newaxis,...]
    prediction = model.predict(X)
    predict_index = (prediction > 0.5).astype(int)
    return

# Metrics calculation
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

# Data augmentation
def augment_minority_class_in_df(df, audio_column, class_column, sr, pitch_shifts):
    # Determine the minority class
    class_counts = df[class_column].value_counts()
    minority_class = class_counts.idxmin()
    
    # Filter
    minority_df = df[df[class_column] == minority_class]

    augmented_rows = []
    for _, row in minority_df.iterrows():
        original_audio = row[audio_column]
        original_class = row[class_column]
        
        # Add the original row
        augmented_rows.append({audio_column: original_audio, class_column: original_class})
        
        # Create augmented samples
        for shift in pitch_shifts:
            augmented_audio = librosa.effects.pitch_shift(y=original_audio, sr=sr, n_steps=shift)
            augmented_rows.append({audio_column: augmented_audio, class_column: original_class})
    
    augmented_df = pd.DataFrame(augmented_rows)
    balanced_df = pd.concat([df, augmented_df], ignore_index=True)

    # Extract MFCC features
    balanced_df['mfcc_features'] = balanced_df[audio_column].apply(lambda audio: extract_mfcc(audio, sr))
    return balanced_df

def main():

    df = pd.read_csv('df_updated_padded.csv')
    print(df['Diagnosis'].value_counts())
    print(df['denoised_audio'].apply(type).value_counts())
    df['denoised_audio'] = df['denoised_audio'].apply(lambda x: np.array(eval(x)))
    balanced_df = augment_minority_class_in_df(df, 'denoised_audio', 'Diagnosis', 8000, [-2, 2])
    print(f"Original dataset size: {len(balanced_df['denoised_audio'][balanced_df['Diagnosis'] == 0])}")
    print(f"Original dataset size: {len(balanced_df['denoised_audio'][balanced_df['Diagnosis'] == 1])}")

    X = np.array(balanced_df['mfcc_features'].tolist())
    y = balanced_df['Diagnosis'].values
    print(X.shape)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(X, y, 0.3, 0.2)

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='hyperparameter_tuning',
        project_name='respiratory_sound_classification'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))
    
    # Evaluate the model
    best_model = tuner.get_best_models(num_models=1)[0]
    test_error, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)    
    print(f'Test Error (Loss): {test_error}')
    print(f'Test Accuracy: {test_accuracy}')

    # Make predictions
    X_sample = X_test[100]  
    y_sample = y_test[100]
    predict(best_model, X_sample, y_sample)

    # Evaluate and calculate metrics
    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    sensitivity, specificity, score, accuracy = calculate_metrics(y_test, y_pred)
    print(f"Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%, Score: {score:.2f}%, Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score: {f1:.2f}")

    # ROC, AUC curve
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict(X_test))
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc_score:.2f})')
    plt.show() 
    
if __name__ == '__main__':
    main()