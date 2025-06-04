import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# Load Processed Data
data = np.load("processed_training_data_with_angles.npy")  # Shape: (num_samples, 6)
X = data[:, :-1]  # Features: |vx|, |vy|, dir_x, dir_y, angle
y = data[:, -1].astype(int)  # Labels

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Model
model = tf.keras.models.load_model("gesture_classification_model.h5")

# Select three features for visualization
feature1_idx, feature2_idx, feature3_idx = 0, 1, 4  # |vx|, |vy|, angle
X_vis = X_test[:, [feature1_idx, feature2_idx, feature3_idx]]

# Define class colors
colors = ['blue', 'orange', 'green', 'red']
class_names = ['Swipe Up', 'Swipe Down', 'Swipe Left', 'Swipe Right']

# Create a 3D Scatter Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Scatter Plot of Test Data
for class_idx in range(4):
    mask = (y_test == class_idx)
    ax.scatter(X_vis[mask, 0], X_vis[mask, 1], X_vis[mask, 2], 
               c=colors[class_idx], label=class_names[class_idx], s=50, alpha=0.7)

# Generate Grid for Decision Boundary
vx_range = np.linspace(X_test[:, feature1_idx].min(), X_test[:, feature1_idx].max(), 20)
vy_range = np.linspace(X_test[:, feature2_idx].min(), X_test[:, feature2_idx].max(), 20)
angle_range = np.linspace(X_test[:, feature3_idx].min(), X_test[:, feature3_idx].max(), 20)

# Create Mesh Grid of Points
VX, VY, ANG = np.meshgrid(vx_range, vy_range, angle_range)
grid_points = np.c_[VX.ravel(), VY.ravel(), ANG.ravel()]

# Prepare 5D input for model by filling missing features with mean values
mean_values = X_test.mean(axis=0)
grid_5D = np.zeros((grid_points.shape[0], X.shape[1]))
grid_5D[:, feature1_idx] = grid_points[:, 0]
grid_5D[:, feature2_idx] = grid_points[:, 1]
grid_5D[:, feature3_idx] = grid_points[:, 2]

for i in range(X.shape[1]):
    if i not in [feature1_idx, feature2_idx, feature3_idx]:
        grid_5D[:, i] = mean_values[i]

# Predict Labels for Grid Points
Z = model.predict(grid_5D)
Z_labels = np.argmax(Z, axis=1)  # Convert predictions to class labels

# Scatter Plot for Decision Boundary
ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], 
           c=np.array(colors)[Z_labels], marker='o', alpha=0.1, s=5)

ax.set_xlabel('|vx| (X velocity magnitude)')
ax.set_ylabel('|vy| (Y velocity magnitude)')
ax.set_zlabel('Angle (degrees)')
ax.set_title('3D Decision Boundary in |vx|, |vy|, and Angle')
ax.legend()
plt.show()