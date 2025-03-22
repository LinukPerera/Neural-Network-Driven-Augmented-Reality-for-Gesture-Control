import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load Processed Data
data = np.load("processed_training_data_with_angles.npy")  # Shape: (num_samples, num_features + 1)
X = data[:, :-1]  # All columns except the last one (features: |velocity|, direction, angle)
y = data[:, -1]   # Last column (filtered labels)

# Ensure labels are integers
y = y.astype(int)  # Labels should already be integers (0, 1, 2, 3) from the first script

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))  # Number of unique classes (should be 4: swipe up/down/left/right)
y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split Data into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),  # Input layer matches number of features (5: |vx|, |vy|, dir_x, dir_y, angle)
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),  # Normalizes activations
    keras.layers.Dropout(0.3),  # Prevents overfitting

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(32, activation="relu"),
    
    keras.layers.Dense(num_classes, activation="softmax")  # Output layer matches number of classes
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate Model on Test Set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save Model
model.save("gesture_classification_model.h5")