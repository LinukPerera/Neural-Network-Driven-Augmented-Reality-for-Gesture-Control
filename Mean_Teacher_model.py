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
student = keras.models.load_model("gesture_classification_model.h5")
# Recompile student model with fresh optimizer
student.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Clone and recompile teacher model
teacher = keras.models.clone_model(student)
teacher.set_weights(student.get_weights())
teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load preprocessed unlabeled data
unlabeled_data = np.load("preprocessed_unlabeled_data.npy")
X_unlabeled = unlabeled_data[:, :-1]

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

# Training parameters
alpha = 0.99  # EMA decay rate
consistency_weight = 0.1
epochs = 150

# Custom training loop
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
for epoch in range(epochs):
    # Train student on labeled data
    metrics = student.fit(X_labeled, y_labeled_onehot, epochs=1, batch_size=16, validation_split=0.2, verbose=0)
    history['accuracy'].append(metrics.history['accuracy'][0])
    history['val_accuracy'].append(metrics.history['val_accuracy'][0])
    history['loss'].append(metrics.history['loss'][0])
    history['val_loss'].append(metrics.history['val_loss'][0])

    # Predict on unlabeled data
    teacher_preds = teacher.predict(X_unlabeled, verbose=0)  # Already NumPy array
    student_preds = student.predict(X_unlabeled, verbose=0)  # Already NumPy array

    # Recoil detection
    recoil_idx = np.zeros(len(X_unlabeled), dtype=bool)
    for i in range(1, len(X_unlabeled)):
        prev_velocity = np.linalg.norm(X_unlabeled[i-1, :2] * [1000, 1000])
        curr_velocity = np.linalg.norm(X_unlabeled[i, :2] * [1000, 1000])
        prev_x_dir, curr_x_dir = X_unlabeled[i-1, 2] * 2 - 1, X_unlabeled[i, 2] * 2 - 1
        if curr_velocity < 300 and curr_velocity < 0.5 * prev_velocity and prev_x_dir != curr_x_dir:
            recoil_idx[i] = True
            teacher_preds[i] = np.array([1, 0, 0, 0])  # Map to "Rest" (adjust if new class)

    # Consistency loss (MSE)
    consistency_loss = np.mean((teacher_preds[~recoil_idx] - student_preds[~recoil_idx]) ** 2)
    student_loss = student.evaluate(X_labeled, y_labeled_onehot, verbose=0)[0] + consistency_weight * consistency_loss
    print(f"Epoch {epoch+1}, Student Loss: {student_loss:.4f}, Consistency Loss: {consistency_loss:.4f}")

    # Update teacher weights
    for w_t, w_s in zip(teacher.weights, student.weights):
        w_t.assign(alpha * w_t + (1 - alpha) * w_s)

# Save student model
student.save("mean_teacher_recoil_model.h5")
print("Mean Teacher model saved as 'mean_teacher_recoil_model.h5'")

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

y_pred_probs = student.predict(X_test_normalized)  # Already NumPy array
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gesture_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix (Mean Teacher)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('150_epoch_mean_teacher_confusion_matrix.png')
print("Confusion matrix saved as 'mean_teacher_confusion_matrix.png'")

# Training Plots
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Mean Teacher Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('150_epoch_mean_teacher_accuracy_plot.png')
print("Accuracy plot saved as 'mean_teacher_accuracy_plot.png'")

plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Mean Teacher Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('150_epoch_mean_teacher_loss_plot.png')
print("Loss plot saved as 'mean_teacher_loss_plot.png'")

# Recoil Visualization
recoil_data = X_unlabeled[recoil_idx]
gesture_data = X_unlabeled[~recoil_idx]
plt.figure(figsize=(10, 6))
plt.scatter(gesture_data[:, 0] * 1000, gesture_data[:, 4] * 360, c='blue', label='Gestures', alpha=0.5)
plt.scatter(recoil_data[:, 0] * 1000, recoil_data[:, 4] * 360, c='red', label='Recoils', alpha=0.5)
plt.title('Recoil vs. Gesture Detection (Mean Teacher)')
plt.xlabel('Absolute X Velocity')
plt.ylabel('Movement Angle (Degrees)')
plt.legend()
plt.grid(True)
plt.savefig('150_epoch_mean_teacher_recoil_plot.png')
print("Recoil plot saved as 'mean_teacher_recoil_plot.png'")