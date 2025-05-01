import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Gesture labels (same as in Wgui.py)
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]

# Load the trained model
model = keras.models.load_model("gesture_classification_model.h5")

# Load the test data
test_data = np.load("processed_testing_data_with_angles.npy")

# Assuming test_data has shape (n_samples, n_features + 1) where last column is the label
# Adjust this based on actual data structure
X_test = test_data[:, :-1]  # Features (velocity and possibly angles)
y_test = test_data[:, -1].astype(int)  # Labels (0 to 4)

# If the model expects 5D input but X_test has fewer dimensions, pad with zeros
if X_test.shape[1] == 2:
    X_test = np.pad(X_test, ((0, 0), (0, 3)), 'constant', constant_values=0)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=gesture_classes))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Optional: Plot distribution of prediction confidences
confidences = np.max(y_pred_probs, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.tight_layout()

# Save the confidence distribution plot
plt.savefig('confidence_distribution.png')
print("Confidence distribution plot saved as 'confidence_distribution.png'")