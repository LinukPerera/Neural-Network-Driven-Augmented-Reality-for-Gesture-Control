import csv
import random

def shuffle_samples(input_file, output_file):
    # Read all lines from the input file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)
    
    # Separate header and data
    header = lines[0]
    data = lines[1:]
    
    # Calculate number of samples
    lines_per_sample = 15
    if len(data) % lines_per_sample != 0:
        raise ValueError("Number of data lines must be divisible by 15")
    
    num_samples = len(data) // lines_per_sample
    
    # Group lines into samples
    samples = []
    for i in range(0, len(data), lines_per_sample):
        sample = data[i:i + lines_per_sample]
        samples.append(sample)
    
    # Shuffle the samples
    random.shuffle(samples)
    
    # Flatten the shuffled samples back into a single list
    shuffled_data = []
    for sample in samples:
        shuffled_data.extend(sample)
    
    # Combine header with shuffled data
    output_lines = [header] + shuffled_data
    
    # Write to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_lines)

# Example usage
try:
    input_file = 'real_world_gesture_data.csv'    # Replace with your input file name
    output_file = 'shuffeled1.csv'  # Replace with your desired output file name
    shuffle_samples(input_file, output_file)
    print("Samples shuffled successfully!")
except ValueError as e:
    print(f"Error: {e}")
except FileNotFoundError:
    print("Error: Input file not found")
except Exception as e:
    print(f"An unexpected error occurred: {e}")