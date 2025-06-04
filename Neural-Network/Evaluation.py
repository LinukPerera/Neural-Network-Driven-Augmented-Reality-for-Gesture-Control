import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
training_data = np.load("processed_training_data_with_angles.npy")
labels = np.load("labels.npy")

# Gesture classes (must match the original script)
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Left", "Swipe Right"]
label_to_gesture = {idx: gesture for idx, gesture in enumerate(gesture_classes)}

# Print raw data
print("=== Raw Data ===")
print("Training Data (Velocities):")
print(training_data)
print("\nLabels:")
print(labels)
print(f"\nTotal samples: {len(training_data)}")

# Organize data by gesture
gesture_data = {gesture: [] for gesture in gesture_classes}
for velocity, label in zip(training_data, labels):
    gesture = label_to_gesture[label]
    gesture_data[gesture].append(velocity)

# 1. Scatter plot of all velocities (x vs y)
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red']  # One color per gesture
for idx, gesture in enumerate(gesture_classes):
    velocities = np.array(gesture_data[gesture])
    if len(velocities) > 0:
        x_velocities = velocities[:, 0]  # dx
        y_velocities = velocities[:, 1]  # dy
        plt.scatter(x_velocities, y_velocities, c=colors[idx], label=gesture, alpha=0.6)

plt.xlabel("X Velocity (pixels/second)")
plt.ylabel("Y Velocity (pixels/second)")
plt.title("Velocity Profiles of All Gestures")
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.show()

# 2. Per-gesture velocity profiles (x and y over sample index)
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
fig.suptitle("Velocity Profiles per Gesture", fontsize=16)

for idx, gesture in enumerate(gesture_classes):
    velocities = np.array(gesture_data[gesture])
    if len(velocities) > 0:
        sample_indices = np.arange(len(velocities))
        x_velocities = velocities[:, 0]  # dx
        y_velocities = velocities[:, 1]  # dy

        # Plot X velocity
        axs[idx].plot(sample_indices, x_velocities, label="X Velocity", color='blue')
        # Plot Y velocity
        axs[idx].plot(sample_indices, y_velocities, label="Y Velocity", color='orange')
        axs[idx].set_title(f"{gesture}")
        axs[idx].set_ylabel("Velocity (pixels/second)")
        axs[idx].legend()
        axs[idx].grid(True)
        axs[idx].axhline(0, color='black', linewidth=0.5, linestyle='--')

axs[-1].set_xlabel("Sample Index")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
plt.show()

# 3. Summary statistics
print("\n=== Summary Statistics ===")
for gesture in gesture_classes:
    velocities = np.array(gesture_data[gesture])
    if len(velocities) > 0:
        avg_x = np.mean(velocities[:, 0])
        avg_y = np.mean(velocities[:, 1])
        std_x = np.std(velocities[:, 0])
        std_y = np.std(velocities[:, 1])
        print(f"\n{gesture}:")
        print(f"  Samples: {len(velocities)}")
        print(f"  Avg X Velocity: {avg_x:.2f} ± {std_x:.2f} pixels/second")
        print(f"  Avg Y Velocity: {avg_y:.2f} ± {std_y:.2f} pixels/second")
    else:
        print(f"\n{gesture}: No samples collected.")