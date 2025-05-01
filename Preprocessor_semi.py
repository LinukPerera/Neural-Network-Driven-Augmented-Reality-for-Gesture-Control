import numpy as np
import pandas as pd

# Load recorded data
data = np.load("unlabeled_gesture_data.npy")

# Filter outliers
max_velocity = 1000  # Based on processed_data_200_with_angles.csv
valid_idx = (data[:, 0] < max_velocity) & (data[:, 1] < max_velocity) & (data[:, 5] >= 0)  # Exclude "Rest"
filtered_data = data[valid_idx]

# Normalize features (scale to [0, 1] based on observed ranges)
X = filtered_data[:, :5].copy()
X[:, 0] = X[:, 0] / max_velocity  # abs_x_velocity
X[:, 1] = X[:, 1] / max_velocity  # abs_y_velocity
X[:, 2] = (X[:, 2] + 1) / 2  # x_direction: [-1, 1] to [0, 1]
X[:, 3] = (X[:, 3] + 1) / 2  # y_direction: [-1, 1] to [0, 1]
X[:, 4] = X[:, 4] / 360  # movement_angle_degrees: [0, 360] to [0, 1]

# Pseudo-labels
y = filtered_data[:, 5].astype(int)

# Save preprocessed data
np.save("preprocessed_unlabeled_data.npy", np.column_stack((X, y)))
print(f"Preprocessed data saved as 'preprocessed_unlabeled_data.npy' with {len(X)} samples.")
