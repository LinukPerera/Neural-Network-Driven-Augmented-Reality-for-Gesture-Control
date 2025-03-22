import numpy as np

# Load the original velocity data and labels
velocities = np.load("training_data.npy")  # Shape: (num_samples, 2)
labels = np.load("labels.npy")  # Assuming you have corresponding labels

# Ensure data is in the correct format
if velocities.ndim != 2 or velocities.shape[1] != 2:
    raise ValueError("Invalid velocity data shape. Expected (num_samples, 2).")

# Separate the X and Y components of the velocities
X_velocities = velocities[:, 0]  # Left (-) / Right (+)
Y_velocities = velocities[:, 1]  # Up (-) / Down (+)

# Create initial mask for all samples
valid_mask = np.ones(len(velocities), dtype=bool)

# Rule 1: Filter based on class-specific velocity directions
# Assuming labels are: 0=swipe up, 1=swipe down, 2=swipe left, 3=swipe right
swipe_up_mask = (labels == 0) & (Y_velocities < 0)  # Negative Y velocity
swipe_down_mask = (labels == 1) & (Y_velocities > 0)  # Positive Y velocity
swipe_left_mask = (labels == 2) & (X_velocities > 0)  # Negative X velocity
swipe_right_mask = (labels == 3) & (X_velocities < 0)  # Positive X velocity

# Combine class-specific masks
class_direction_mask = swipe_up_mask | swipe_down_mask | swipe_left_mask | swipe_right_mask
valid_mask &= class_direction_mask

# Rule 2: Ignore samples past 1000 in either x or y direction
extreme_value_mask = (np.abs(X_velocities) <= 1000) & (np.abs(Y_velocities) <= 1000)
valid_mask &= extreme_value_mask

# Rule 3: Ignore samples where BOTH x AND y velocities are below 150
low_velocity_mask = ~((np.abs(X_velocities) < 150) & (np.abs(Y_velocities) < 150))
valid_mask &= low_velocity_mask

# Apply the mask to filter the data
filtered_velocities = velocities[valid_mask]
filtered_labels = labels[valid_mask]

# Compute the absolute velocities
abs_velocities = np.abs(filtered_velocities)

# Compute the movement direction (+1 for right/up, -1 for left/down)
direction_labels = np.sign(filtered_velocities)

# Compute the angle of movement (in degrees)
angles = np.arctan2(filtered_velocities[:, 1], filtered_velocities[:, 0])
angles_degrees = np.degrees(angles)
angles_degrees = (angles_degrees + 360) % 360

# Handle zero-velocity cases
zero_mask = np.linalg.norm(filtered_velocities, axis=1) < 1e-6
direction_labels[zero_mask] = 0

# Combine all features
processed_data = np.hstack((abs_velocities, 
                          direction_labels, 
                          angles_degrees.reshape(-1, 1),
                          filtered_labels.reshape(-1, 1)))

# Save the processed data
np.save("processed_training_data_with_angles.npy", processed_data)

print(f"Processed data saved! Shape: {processed_data.shape}")
print(f"Original samples: {len(velocities)}, Filtered samples: {len(filtered_velocities)}")
print(f"Removed {len(velocities) - len(filtered_velocities)} anomalous samples")