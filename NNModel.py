import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Gesture labels (based on 4 classes, excluding "Rest" which is handled separately in Wgui.py)
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]

# Load Processed Data
data = np.load("processed_data_200_with_angles.npy")  # Shape: (num_samples, num_features + 1)
X = data[:, :-1]  # All columns except the last one (features: |velocity|, direction, angle)
y = data[:, -1]   # Last column (filtered labels)

# Ensure labels are integers
y = y.astype(int)  # Labels should be integers (0, 1, 2, 3)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))  # Number of unique classes (should be 4)
y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split Data into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),  # Input layer matches number of features
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(32, activation="relu"),
    
    keras.layers.Dense(num_classes, activation="softmax")  # Output layer matches number of classes
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2)

# Evaluate Model on Test Set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save Model
model.save("gesture_classification_model.h5")

# --- Additional Visualizations and Statistics ---

# 1. Plot Training & Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
print("Accuracy plot saved as 'accuracy_plot.png'")

# 2. Plot Training & Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
print("Loss plot saved as 'loss_plot.png'")

# 3. Confusion Matrix and Classification Report
# Get predictions for test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=gesture_classes))

# 4. Histogram of Prediction Confidences
confidences = np.max(y_pred_probs, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('confidence_distribution.png')
print("Confidence distribution plot saved as 'confidence_distribution.png'")
