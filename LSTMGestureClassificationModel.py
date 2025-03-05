#LSTM Classification Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.02),
             recurrent_dropout=0.4),
        BatchNormalization(),
        Dropout(0.6),

        LSTM(16, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(0.02),
             recurrent_dropout=0.4),
        BatchNormalization(),
        Dropout(0.6),

        Dense(8, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Enhanced data augmentation
def augment_data(X, noise_factor=0.05, time_shift=2):
    noise = np.random.normal(0, noise_factor, X.shape)
    X_noisy = X + noise
    
    for i in range(X.shape[0]):
        shift = np.random.randint(-time_shift, time_shift + 1)
        if shift != 0:
            if shift > 0:
                X_noisy[i, :-shift] = X_noisy[i, shift:]
                X_noisy[i, -shift:] = 0
            else:
                X_noisy[i, -shift:] = X_noisy[i, :shift]
                X_noisy[i, :shift] = 0
    return X_noisy

# Get input shape and number of classes
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = 2

# Create model
model = create_lstm_model(input_shape, num_classes)

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=10,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Train model with data augmentation
history = model.fit(
    augment_data(X_train), y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[learning_rate_reduction, early_stopping]
)

# Save model and plot
model.save('gesture_lstm_model.h5')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
