import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed data
data = np.load("processed_training_data_with_angles.npy")

# Extract components
X_velocities = data[:, 0] * data[:, 2]  # abs_x * direction_x to restore original signed values
Y_velocities = data[:, 1] * data[:, 3]  # abs_y * direction_y to restore original signed values
labels = data[:, 5]  # Class labels (0-3)
angles = data[:, 4]  # Angles in degrees

# Define class names and colors
class_names = ['Swipe Up', 'Swipe Down', 'Swipe Left', 'Swipe Right']
colors = ['blue', 'red', 'green', 'purple']

# Create scatter plot
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(X_velocities[mask], Y_velocities[mask], 
                c=colors[int(label)], 
                label=class_names[int(label)], 
                alpha=0.5, 
                s=50)

# Add plot details
plt.xlabel('X Velocity (Left - / Right +)')
plt.ylabel('Y Velocity (Up - / Down +)')
plt.title('Preprocessed Swipe Data by Class')
plt.legend()
plt.grid(True, alpha=0.3)

# Set reasonable axis limits based on our filtering (1000 max)
plt.xlim(-1100, 1100)
plt.ylim(-1100, 1100)

# Add origin lines
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Show plot
plt.show()

# Print some basic statistics
print("Samples per class:")
for i, name in enumerate(class_names):
    count = np.sum(labels == i)
    print(f"{name}: {count} samples")