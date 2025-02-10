import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load model
model = tf.keras.models.load_model('gesture_lstm_model.h5')

# Get unique classes from the test labels
unique_classes = np.unique(y_test)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_classes, 
    target_names=[str(cls) for cls in unique_classes]))

# Confusion matrix visualization
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', 
    xticklabels=[str(cls) for cls in unique_classes],
    yticklabels=[str(cls) for cls in unique_classes])
plt.title('Gesture Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')