import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Gesture classes (matches semi-supervised models)
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]

# Load test data
test_data = np.load("processed_testing_data_with_angles.npy")
X_test = test_data[:, :-1]  # Features: abs_x_velocity, abs_y_velocity, x_direction, y_direction, movement_angle_degrees
y_test = test_data[:, -1].astype(int)  # Labels: 0-3

# Normalize test data (same as training)
X_test_normalized = X_test.copy()
X_test_normalized[:, 0] = X_test_normalized[:, 0] / 1000  # abs_x_velocity
X_test_normalized[:, 1] = X_test_normalized[:, 1] / 1000  # abs_y_velocity
X_test_normalized[:, 2] = (X_test_normalized[:, 2] + 1) / 2  # x_direction: [-1, 1] to [0, 1]
X_test_normalized[:, 3] = (X_test_normalized[:, 3] + 1) / 2  # y_direction: [-1, 1] to [0, 1]
X_test_normalized[:, 4] = X_test_normalized[:, 4] / 360  # movement_angle_degrees

# Load models
self_training_model = keras.models.load_model("self_training_recoil_model.h5")
mean_teacher_model = keras.models.load_model("mean_teacher_recoil_model.h5")

# Function to evaluate a model and return metrics
def evaluate_model(model, model_name, X_test, y_test):
    # Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=gesture_classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, accuracy, report, cm

# Evaluate both models
self_training_pred, self_training_accuracy, self_training_report, self_training_cm = evaluate_model(
    self_training_model, "Self-Training", X_test_normalized, y_test
)
mean_teacher_pred, mean_teacher_accuracy, mean_teacher_report, mean_teacher_cm = evaluate_model(
    mean_teacher_model, "Mean Teacher", X_test_normalized, y_test
)

# Print classification reports
print("\n=== Self-Training Model Evaluation ===")
print(f"Overall Accuracy: {self_training_accuracy:.4f}")
print("\nClassification Report:")
print(pd.DataFrame(self_training_report).transpose().round(4))

print("\n=== Mean Teacher Model Evaluation ===")
print(f"Overall Accuracy: {mean_teacher_accuracy:.4f}")
print("\nClassification Report:")
print(pd.DataFrame(mean_teacher_report).transpose().round(4))

# Plot confusion matrices
plt.figure(figsize=(10, 8))
sns.heatmap(self_training_cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix (Self-Training)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('self_training_confusion_matrix_test.png')
plt.close()
print("Self-Training confusion matrix saved as 'self_training_confusion_matrix_test.png'")

plt.figure(figsize=(10, 8))
sns.heatmap(mean_teacher_cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.title('Confusion Matrix (Mean Teacher)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('mean_teacher_confusion_matrix_test.png')
plt.close()
print("Mean Teacher confusion matrix saved as 'mean_teacher_confusion_matrix_test.png'")

# Plot per-class metrics for each model
def plot_metrics(report, model_name):
    metrics_df = pd.DataFrame({
        'Precision': [report[cls]['precision'] for cls in gesture_classes],
        'Recall': [report[cls]['recall'] for cls in gesture_classes],
        'F1-Score': [report[cls]['f1-score'] for cls in gesture_classes]
    }, index=gesture_classes)
    
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f'{model_name} Per-Class Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_metrics_plot.png')
    plt.close()

plot_metrics(self_training_report, "Self-Training")
print("Self-Training metrics plot saved as 'self_training_metrics_plot.png'")
plot_metrics(mean_teacher_report, "Mean Teacher")
print("Mean Teacher metrics plot saved as 'mean_teacher_metrics_plot.png'")

# Comparative plot: Accuracy and Macro F1-Score
comparison_df = pd.DataFrame({
    'Model': ['Self-Training', 'Mean Teacher'],
    'Accuracy': [self_training_accuracy, mean_teacher_accuracy],
    'Macro F1-Score': [self_training_report['macro avg']['f1-score'], mean_teacher_report['macro avg']['f1-score']]
})

comparison_df.plot(x='Model', kind='bar', figsize=(10, 6))
plt.title('Model Comparison: Accuracy and Macro F1-Score')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.savefig('model_comparison_plot.png')
plt.close()
print("Comparative plot saved as 'model_comparison_plot.png'")