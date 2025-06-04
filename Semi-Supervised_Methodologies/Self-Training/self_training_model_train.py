import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)

# Gesture classes
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]

# Load existing model
model = keras.models.load_model("gesture_classification_model.h5")
# Recompile model with fresh optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load preprocessed unlabeled data
unlabeled_data = np.load("preprocessed_unlabeled_data.npy")
X_unlabeled = unlabeled_data[:, :-1]
y_unlabeled = unlabeled_data[:, -1].astype(int)

# Load original labeled data
labeled_data = pd.read_csv("processed_data_200_with_angles.csv")
X_labeled = labeled_data[["abs_x_velocity", "abs_y_velocity", "x_direction", "y_direction", "movement_angle_degrees"]].values
y_labeled = labeled_data["label"].astype(int)
y_labeled_onehot = keras.utils.to_categorical(y_labeled, num_classes=4)

# Normalize labeled data
X_labeled = X_labeled.copy()
X_labeled[:, 0] = X_labeled[:, 0] / 1000
X_labeled[:, 1] = X_labeled[:, 1] / 1000
X_labeled[:, 2] = (X_labeled[:, 2] + 1) / 2
X_labeled[:, 3] = (X_labeled[:, 3] + 1) / 2
X_labeled[:, 4] = X_labeled[:, 4] / 360

# Pseudo-labeling with recoil filtering
pseudo_probs = model.predict(X_unlabeled)  # Already NumPy array
pseudo_labels = np.argmax(pseudo_probs, axis=1)
confidence_scores = np.max(pseudo_probs, axis=1)

# Recoil detection: low velocity and opposite direction
recoil_idx = np.zeros_like(confidence_scores, dtype=bool)
for i in range(1, len(X_unlabeled)):
    prev_velocity = np.linalg.norm(X_unlabeled[i-1, :2] * [1000, 1000])  # Denormalize
    curr_velocity = np.linalg.norm(X_unlabeled[i, :2] * [1000, 1000])
    prev_x_dir, curr_x_dir = X_unlabeled[i-1, 2] * 2 - 1, X_unlabeled[i, 2] * 2 - 1
    if curr_velocity < 300 and curr_velocity < 0.5 * prev_velocity and prev_x_dir != curr_x_dir:
        recoil_idx[i] = True
        pseudo_labels[i] = -1  # Mark as "Rest"

# Filter high-confidence pseudo-labels (exclude recoils)
confidence_threshold = 0.95
high_conf_idx = (confidence_scores > confidence_threshold) & (~recoil_idx)
X_pseudo = X_unlabeled[high_conf_idx]
y_pseudo = keras.utils.to_categorical(pseudo_labels[high_conf_idx], num_classes=4)

# Combine datasets
X_combined = np.vstack((X_labeled, X_pseudo))
y_combined = np.vstack((y_labeled_onehot, y_pseudo))

# Retrain model
history = model.fit(X_combined, y_combined, epochs=200, batch_size=16, validation_split=0.2)

# Save model
model.save("self_training_recoil_model.h5")
print("Self-training model saved as 'self_training_recoil_model.h5'")

# Evaluate on test set
test_data = np.load("processed_testing_data_with_angles.npy")
X_test = test_data[:, :-1]
y_test = test_data[:, -1].astype(int)
X_test_normalized = X_test.copy()
X_test_normalized[:, 0] = X_test_normalized[:, 0] / 1000
X_test_normalized[:, 1] = X_test_normalized[:, 1] / 1000
X_test_normalized[:, 2] = (X_test_normalized[:, 2] + 1) / 2
X_test_normalized[:, 3] = (X_test_normalized[:, 3] + 1) / 2
X_test_normalized[:, 4] = X_test_normalized[:, 4] / 360

y_pred_probs = model.predict(X_test_normalized)  # Already NumPy array
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gesture_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix (Self-Training)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('200_epoch_self_training_confusion_matrix.png')
print("Confusion matrix saved as 'self_training_confusion_matrix.png'")

# Training Plots
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Self-Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('200_epoch_self_training_accuracy_plot.png')
print("Accuracy plot saved as 'self_training_accuracy_plot.png'")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Self-Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('200_epoch_self_training_loss_plot.png')
print("Loss plot saved as 'self_training_loss_plot.png'")

# Recoil Visualization
recoil_data = X_unlabeled[recoil_idx]
gesture_data = X_unlabeled[~recoil_idx]
plt.figure(figsize=(10, 6))
plt.scatter(gesture_data[:, 0] * 1000, gesture_data[:, 4] * 360, c='blue', label='Gestures', alpha=0.5)
plt.scatter(recoil_data[:, 0] * 1000, recoil_data[:, 4] * 360, c='red', label='Recoils', alpha=0.5)
plt.title('Recoil vs. Gesture Detection (Self-Training)')
plt.xlabel('Absolute X Velocity')
plt.ylabel('Movement Angle (Degrees)')
plt.legend()
plt.grid(True)
plt.savefig('200_epoch_self_training_recoil_plot.png')
print("Recoil plot saved as 'self_training_recoil_plot.png'")
