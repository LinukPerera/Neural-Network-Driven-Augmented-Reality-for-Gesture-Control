import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Parameters
INPUT_DIR = 'gesture_data'  # Update this to your actual gesture_data path, e.g., '/Users/linukperera/gesture_data'
OUTPUT_DIR = 'velocity_profiles'
GESTURE_CLASSES = ['Swipe Up', 'Swipe Down', 'Swipe Right', 'Swipe Left']
NUM_SAMPLES = 20  # Samples per gesture
NUM_FRAMES = 30   # Frames per sample

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def validate_input_dir():
    """Check if INPUT_DIR exists and contains CSV files."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' does not exist.")
        print("Please update INPUT_DIR to the correct path and try again.")
        exit(1)
    
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{INPUT_DIR}'.")
        print("Please ensure the directory contains CSV files with the format '<gesture>_sample_<number>_<timestamp>.csv'.")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files in '{INPUT_DIR}'.")

def load_gesture_data(gesture):
    """Load all samples for a gesture and return x_vel and y_vel arrays."""
    x_vels = []
    y_vels = []
    found_files = []
    
    # Find all CSV files for the gesture
    for csv_file in os.listdir(INPUT_DIR):
        if csv_file.startswith(gesture) and csv_file.endswith('.csv'):
            file_path = os.path.join(INPUT_DIR, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # Verify required columns
                required_columns = ['x_vel', 'y_vel']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: Skipping '{csv_file}' - missing columns: {set(required_columns) - set(df.columns)}")
                    continue
                
                # Ensure 30 frames
                if len(df) < NUM_FRAMES:
                    last_row = df.iloc[-1:].copy()
                    for _ in range(NUM_FRAMES - len(df)):
                        df = pd.concat([df, last_row], ignore_index=True)
                elif len(df) > NUM_FRAMES:
                    df = df.iloc[:NUM_FRAMES]
                
                x_vels.append(df['x_vel'].values)
                y_vels.append(df['y_vel'].values)
                found_files.append(csv_file)
            
            except Exception as e:
                print(f"Warning: Failed to process '{csv_file}' - {str(e)}")
                continue
    
    # Convert to numpy arrays and compute mean
    if not x_vels:
        print(f"Error: No valid samples found for '{gesture}'.")
        return None, None
    
    x_vels = np.array(x_vels)  # Shape: (num_samples, 30)
    y_vels = np.array(y_vels)  # Shape: (num_samples, 30)
    
    # Check number of samples
    if x_vels.shape[0] != NUM_SAMPLES:
        print(f"Warning: Found {x_vels.shape[0]} samples for '{gesture}', expected {NUM_SAMPLES}")
        print(f"Files used: {found_files}")
    
    # Compute average velocity profiles
    avg_x_vel = np.mean(x_vels, axis=0)  # Shape: (30,)
    avg_y_vel = np.mean(y_vels, axis=0)  # Shape: (30,)
    
    return avg_x_vel, avg_y_vel

def plot_and_save_velocity_profile(gesture, avg_x_vel, avg_y_vel):
    """Plot and save the average velocity profile for a gesture."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot x_vel
    ax1.plot(range(NUM_FRAMES), avg_x_vel, 'b-', label='Average X-Velocity')
    ax1.set_ylabel('X-Velocity (pixels/second)')
    ax1.set_title(f'{gesture} Average Velocity Profile')
    ax1.grid(True)
    ax1.legend()
    
    # Plot y_vel
    ax2.plot(range(NUM_FRAMES), avg_y_vel, 'r-', label='Average Y-Velocity')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y-Velocity (pixels/second)')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f'{gesture.replace(" ", "_")}_velocity_profile_{timestamp}.png')
    plt.savefig(filename)
    print(f"Saved: {filename}")
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    print("Generating average velocity profiles for gestures...")
    
    # Validate input directory
    validate_input_dir()
    
    for gesture in GESTURE_CLASSES:
        print(f"\nProcessing {gesture}...")
        avg_x_vel, avg_y_vel = load_gesture_data(gesture)
        if avg_x_vel is None or avg_y_vel is None:
            print(f"Skipping plot for '{gesture}' due to no valid data.")
            continue
        plot_and_save_velocity_profile(gesture, avg_x_vel, avg_y_vel)
    
    print("\nAll velocity profiles generated and saved!")

if __name__ == "__main__":
    main()