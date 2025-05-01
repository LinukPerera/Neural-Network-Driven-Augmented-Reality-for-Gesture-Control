import numpy as np
import pandas as pd

# Load the processed data
data = np.load('processed_data_200_with_angles.npy')

# Create column headers based on the Preprocessor.py structure
column_headers = [
    'abs_x_velocity',
    'abs_y_velocity',
    'x_direction',
    'y_direction',
    'movement_angle_degrees',
    'label'
]

# Convert to pandas DataFrame for easy CSV export
df = pd.DataFrame(data, columns=column_headers)

# Map the numeric labels to descriptive names
label_map = {
    0: 'swipe_up',
    1: 'swipe_down',
    2: 'swipe_left',
    3: 'swipe_right'
}
df['label_description'] = df['label'].map(label_map)

# Save to CSV
output_file = 'processed_data_200_with_angles.csv'
df.to_csv(output_file, index=False)

print(f"Data successfully saved to {output_file}")
print(f"Data shape: {data.shape}")
print("Column descriptions:")
print("- abs_x_velocity: Absolute X velocity value")
print("- abs_y_velocity: Absolute Y velocity value")
print("- x_direction: -1 (left), 0 (none), +1 (right)")
print("- y_direction: -1 (down), 0 (none), +1 (up)")
print("- movement_angle_degrees: Angle in degrees (0-360)")
print("- label: Numeric label (0-3)")
print("- label_description: Text description of the label")
