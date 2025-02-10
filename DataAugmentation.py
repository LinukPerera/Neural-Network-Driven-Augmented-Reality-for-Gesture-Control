#DataAugmentation.py
import csv

def augment_gesture_data(input_csv, output_csv):
    # Spatial shift configurations (adjust deltas as needed)
    shifts = [
        # Format: (shift_name, x_delta, y_delta)
        ('middle_left', -0.15, 0),
        ('middle_right', +0.15, 0),
        ('top_center', 0, +0.15),
        ('top_left', -0.15, +0.15),
        ('top_right', +0.15, +0.15),
        ('bottom_center', 0, -0.15),
        ('bottom_left', -0.15, -0.15),
        ('bottom_right', +0.15, -0.15),
    ]

    # Scaling factors (comment out problematic scales)
    scales = [
        0.8,   # Smaller hand (further away)
        1.0,   # Original size
        1.2    # Larger hand (closer)
    ]

    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            # Write original sample
            writer.writerow(row)

            label = row[0]
            original = list(map(float, row[2:]))  # Skip label and timestamp

            # Apply spatial shifts
            for shift_name, dx, dy in shifts:
                # Skip shifts that might be problematic for certain gestures
                if "swipe_left" in label and "left" in shift_name:
                    continue
                if "swipe_right" in label and "right" in shift_name:
                    continue

                shifted = []
                for i in range(0, len(original), 3):
                    x = original[i] + dx
                    y = original[i + 1] + dy
                    z = original[i + 2]
                    
                    # Clamp values between 0-1
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    
                    shifted.extend([x, y, z])
                
                # Apply scaling to shifted data
                for scale in scales:
                    scaled = []
                    for i in range(0, len(shifted), 3):
                        x = shifted[i] * scale
                        y = shifted[i + 1] * scale
                        z = shifted[i + 2]
                        
                        # Re-clamp scaled values
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        
                        scaled.extend([x, y, z])
                    
                    # Write augmented sample
                    writer.writerow([label, row[1]] + scaled)

if __name__ == "__main__":
    augment_gesture_data('gesture_data.csv', 'augmented_gesture_data.csv')