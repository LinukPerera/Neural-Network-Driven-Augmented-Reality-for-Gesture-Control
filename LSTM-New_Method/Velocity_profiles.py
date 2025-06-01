import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import signal

# Parameters
INPUT_DIR = '/Users/linukperera/Programming/Python/Lv 06 Project/May - 5/gesture_data'  
OUTPUT_DIR = 'velocity_profiles_processed'
TRIMMED_DIR = 'gesture_data_trimmed'  # Directory for trimmed CSVs
GESTURE_CLASSES = ['Swipe Left']
GESTURE_VARIATIONS = {
    #'Swipe Up': ['Swipe_Up', 'SwipeUp', 'swipe_up', 'Swipe Up', 'swipe-up', 'SWIPE_UP', 'Swipe-Up'],
    #'Swipe Down': ['Swipe_Down', 'SwipeDown', 'swipe_down', 'Swipe Down', 'swipe-down', 'SWIPE_DOWN', 'Swipe-Down'],
    #'Swipe Right': ['Swipe_Right', 'SwipeRight', 'swipe_right', 'Swipe Right', 'swipe-right', 'SWIPE_RIGHT', 'Swipe-Right'],
    'Swipe Left': ['Swipe_Left', 'SwipeLeft', 'swipe_left', 'Swipe Left', 'swipe-left', 'SWIPE_LEFT', 'Swipe-Left']
}
NUM_SAMPLES = 20  # Samples per gesture
NUM_FRAMES = 15   # Trimmed frames (5 to 20)
FRAME_START = 5   # Start at frame 5
FRAME_END = 20    # End at frame 20

# Create output directories
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(TRIMMED_DIR):
    os.makedirs(TRIMMED_DIR)

def timeout_handler(signum, frame):
    """Handle timeout during file reading."""
    raise TimeoutError("File reading timed out")

def validate_input_dir():
    """Check if INPUT_DIR exists and list all CSV files."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' does not exist.")
        print("Please update INPUT_DIR to the correct path and try again.")
        exit(1)
    
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{INPUT_DIR}'.")
        print("Please ensure the directory contains CSV files with gesture data.")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files in '{INPUT_DIR}':")
    file_gestures = []
    for csv_file in sorted(csv_files):
        assigned_gesture = None
        for gesture, variations in GESTURE_VARIATIONS.items():
            if any(variation.lower() in csv_file.lower() for variation in variations):
                assigned_gesture = gesture
                break
        file_gestures.append((csv_file, assigned_gesture))
    
    for csv_file, gesture in file_gestures:
        print(f"  {csv_file}: Assigned to {gesture if gesture else 'None'}")
    
    # Summarize gesture counts
    gesture_counts = {}
    for gesture in GESTURE_CLASSES:
        gesture_counts[gesture] = sum(1 for _, g in file_gestures if g == gesture)
    print("\nGesture file counts:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count} files")
    
    return csv_files

def load_and_trim_gesture_data(gesture):
    """Load samples, trim to frames 5-20, save trimmed CSVs, and return x_vel and y_vel arrays."""
    x_vels = []
    y_vels = []
    found_files = []
    variations = GESTURE_VARIATIONS[gesture]
    
    # Set up timeout for file reading
    signal.signal(signal.SIGALRM, timeout_handler)
    
    for csv_file in os.listdir(INPUT_DIR):
        if any(variation.lower() in csv_file.lower() for variation in variations) and csv_file.endswith('.csv'):
            file_path = os.path.join(INPUT_DIR, csv_file)
            print(f"Processing file: {csv_file}")
            try:
                # Set 5-second timeout
                signal.alarm(5)
                df = pd.read_csv(file_path)
                signal.alarm(0)
                
                # Verify required columns
                required_columns = ['frame', 'x_vel', 'y_vel', 'angle', 'gesture']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: Skipping '{csv_file}' - missing columns: {set(required_columns) - set(df.columns)}")
                    continue
                
                # Ensure at least 20 frames to trim 5-20
                if len(df) < FRAME_END:
                    last_row = df.iloc[-1:].copy()
                    for _ in range(FRAME_END - len(df)):
                        df = pd.concat([df, last_row], ignore_index=True)
                elif len(df) > FRAME_END:
                    df = df.iloc[:FRAME_END]
                
                # Trim to frames 5-20 (indices 5 to 20, 15 frames)
                df_trimmed = df.iloc[FRAME_START:FRAME_END].copy()
                
                # Adjust frame numbers to start from 0 (0 to 14)
                df_trimmed['frame'] = range(NUM_FRAMES)
                
                # Save trimmed CSV
                trimmed_filename = os.path.join(TRIMMED_DIR, csv_file)
                df_trimmed.to_csv(trimmed_filename, index=False)
                print(f"  Saved trimmed CSV: {trimmed_filename}")
                
                x_vels.append(df_trimmed['x_vel'].values)
                y_vels.append(df_trimmed['y_vel'].values)
                found_files.append(csv_file)
                print(f"  Successfully processed: {csv_file}")
            
            except TimeoutError:
                print(f"Warning: Skipping '{csv_file}' - reading timed out")
                signal.alarm(0)
                continue
            except Exception as e:
                print(f"Warning: Failed to process '{csv_file}' - {str(e)}")
                continue
    
    if not x_vels:
        print(f"Error: No valid samples found for '{gesture}'. Found files: {found_files}")
        return None, None
    
    x_vels = np.array(x_vels)  # Shape: (num_samples, 15)
    y_vels = np.array(y_vels)  # Shape: (num_samples, 15)
    
    print(f"Found {x_vels.shape[0]} samples for '{gesture}': {found_files}")
    if x_vels.shape[0] != NUM_SAMPLES:
        print(f"Warning: Expected {NUM_SAMPLES} samples for '{gesture}', found {x_vels.shape[0]}")
    
    avg_x_vel = np.mean(x_vels, axis=0)  # Shape: (15,)
    avg_y_vel = np.mean(y_vels, axis=0)  # Shape: (15,)
    
    return avg_x_vel, avg_y_vel

def plot_and_save_velocity_profile(gesture, avg_x_vel, avg_y_vel):
    """Plot and save the average velocity profile for frames 5-20."""
    print(f"Generating plot for '{gesture}'...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot x_vel for frames 5-20 (labeled as 0-14 for simplicity)
    frames = range(NUM_FRAMES)
    ax1.plot(frames, avg_x_vel, 'b-', label='Average X-Velocity')
    ax1.set_ylabel('X-Velocity (pixels/second)')
    ax1.set_title(f'{gesture} Average Velocity Profile (Frames 5-20)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot y_vel
    ax2.plot(frames, avg_y_vel, 'r-', label='Average Y-Velocity')
    ax2.set_xlabel('Frame (0 = Frame 5)')
    ax2.set_ylabel('Y-Velocity (pixels/second)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f'{gesture.replace(" ", "_")}_velocity_profile_frames_5_20_{timestamp}.png')
    plt.savefig(filename)
    print(f"Saved: {filename}")
    
    plt.show()
    plt.close(fig)

def main():
    print("Generating average velocity profiles for gestures (frames 5-20) and saving trimmed CSVs...")
    
    csv_files = validate_input_dir()
    
    for gesture in GESTURE_CLASSES:
        print(f"\nProcessing {gesture}...")
        avg_x_vel, avg_y_vel = load_and_trim_gesture_data(gesture)
        if avg_x_vel is None or avg_y_vel is None:
            print(f"Skipping plot for '{gesture}' due to no valid data.")
            continue
        plot_and_save_velocity_profile(gesture, avg_x_vel, avg_y_vel)
    
    print("\nAll velocity profiles generated and trimmed CSVs saved!")

if __name__ == "__main__":
    main()