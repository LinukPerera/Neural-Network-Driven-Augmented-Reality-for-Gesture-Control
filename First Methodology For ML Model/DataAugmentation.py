#DataAugmentation.py
import csv

def augment_gesture_data(input_csv, output_csv, sequence_length=15):
    shifts = [
        ('middle_left', -0.15, 0),
        ('middle_right', +0.15, 0),
        ('top_center', 0, +0.15),
        ('top_left', -0.15, +0.15),
        ('top_right', +0.15, +0.15),
        ('bottom_center', 0, -0.15),
        ('bottom_left', -0.15, -0.15),
        ('bottom_right', +0.15, -0.15),
    ]
    scales = [0.8, 1.0, 1.2]

    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        header.insert(1, "sample_id")
        header.insert(2, "frame_index")
        writer.writerow(header)  

        sample_id = 1
        buffer = []

        for row in reader:
            if len(row) < 4:  # Ensure valid row structure
                print(f"Skipping invalid row: {row}")
                continue

            label = row[0]
            timestamp = row[1]  # Keep timestamp for better LTSM
            try:
                landmarks = list(map(float, row[2:]))
            except ValueError:
                print(f"Skipping row with non-numeric values: {row}")
                continue

            if len(landmarks) % 3 != 0:
                print(f"Skipping row with incorrect landmark count: {row}")
                continue

            buffer.append([label, timestamp] + landmarks)

            if len(buffer) == sequence_length:
                augmented_samples = []
                augmented_samples.append((sample_id, label, buffer.copy()))

                for shift_name, dx, dy in shifts:
                    shifted = []
                    for frame in buffer:
                        timestamp = frame[1]
                        original = frame[2:]
                        shifted_frame = []
                        for i in range(0, len(original), 3):
                            x = original[i] + dx
                            y = original[i + 1] + dy
                            z = original[i + 2]
                            x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
                            shifted_frame.extend([x, y, z])
                        shifted.append([label, timestamp] + shifted_frame)

                    for scale in scales:
                        scaled = []
                        for frame in shifted:
                            timestamp = frame[1]
                            original = frame[2:]
                            scaled_frame = []
                            for i in range(0, len(original), 3):
                                x, y, z = original[i] * scale, original[i + 1] * scale, original[i + 2]
                                x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
                                scaled_frame.extend([x, y, z])
                            scaled.append([label, timestamp] + scaled_frame)
                        sample_id += 1
                        augmented_samples.append((sample_id, label, scaled.copy()))

                for sample in augmented_samples:
                    unique_sample_id, label, frames = sample
                    for frame_index, frame in enumerate(frames, start=1):
                        writer.writerow([label, unique_sample_id, frame_index] + frame[1:])  # ✅ Fix writing

                sample_id += 1
                buffer.clear()

    print(f"Data successfully written to {output_csv}")

if __name__ == "__main__":
    augment_gesture_data('gesture_data1.csv', 'real_world_gesture_data.csv')
