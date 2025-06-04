import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Left", "Swipe Right"]
gesture_to_label = {gesture: idx for idx, gesture in enumerate(gesture_classes)}

# Define weights for velocity computation
WEIGHTS = {
    "wrist": 1.0,
    "fingertip": 3.0
}

# Fingertip landmarks (index, middle, ring, pinky)
FINGERTIP_IDS = [8, 12, 16, 20]

def countdown_timer(message, duration=3):
    """ Displays a countdown before recording. """
    for i in range(duration, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"{message} {i}", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("Gesture Recording", frame_copy)
        cv2.waitKey(1000)  # Wait for 1 second per countdown step

def collect_gesture_data(gesture_name, samples_needed=10):
    """ Records weighted gesture velocity over 1 second per sample. """
    velocities = []
    label = gesture_to_label[gesture_name]

    print(f"\nRecording {gesture_name} ({samples_needed} samples needed)")
    
    for sample_count in range(samples_needed):
        print(f"\nGet Ready for {gesture_name} Sample {sample_count + 1}...")

        countdown_timer("Get Ready", 3)  # Display countdown

        print("Recording...")
        sample_velocities = []
        prev_positions = {}
        prev_time = time.time()
        start_time = prev_time

        while time.time() - start_time < 1:  # Record for 1 second
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            current_time = time.time()

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    h, w, _ = frame.shape

                    # Compute weighted velocity using wrist and fingertips
                    velocities_list = []
                    total_weight = 0

                    for idx in [0] + FINGERTIP_IDS:  # Wrist (id=0) and fingertips
                        landmark = hand_landmarks.landmark[idx]
                        curr_pos = np.array([landmark.x * w, landmark.y * h])

                        weight = WEIGHTS["wrist"] if idx == 0 else WEIGHTS["fingertip"]

                        if idx in prev_positions and prev_time is not None:
                            delta_pos = curr_pos - prev_positions[idx]
                            delta_time = current_time - prev_time
                            velocity = (delta_pos / delta_time) * weight
                            velocities_list.append(velocity)
                            total_weight += weight

                        prev_positions[idx] = curr_pos

                    prev_time = current_time

                    # Compute weighted average velocity
                    if velocities_list:
                        weighted_velocity = np.sum(velocities_list, axis=0) / total_weight
                        sample_velocities.append(weighted_velocity)

            # Display status
            cv2.putText(frame, f"Recording {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Gesture Recording", frame)
            cv2.waitKey(1)

        print(f"Sample {sample_count + 1} recorded.")

        # Store the sample's average velocity (with direction)
        if sample_velocities:
            avg_velocity = np.mean(sample_velocities, axis=0)
            velocities.append(avg_velocity)  # Preserve negative values

        # Wait for 2 seconds before next sample
        print("Re-adjust yourself...")
        time.sleep(2)

    return velocities if velocities else []

# Collect training data
training_data = []
labels = []
samples_per_gesture = 50

print("Welcome to Automated Gesture Recording!")
print("Perform gestures naturally. Each sample will be recorded for 1 second with a 2-second transition period.")

for gesture in gesture_classes:
    print(f"\nStarting {gesture}...")
    gesture_velocities = collect_gesture_data(gesture, samples_needed=samples_per_gesture)
    if gesture_velocities:
        training_data.extend(gesture_velocities)
        labels.extend([gesture_to_label[gesture]] * len(gesture_velocities))
        print(f"Collected {len(gesture_velocities)} samples for {gesture}.")
    else:
        print(f"Warning: No valid samples collected for {gesture}.")

# Save data
np.save("training_data.npy", np.array(training_data))
np.save("labels.npy", np.array(labels))
print(f"\nTraining data saved! Recorded {len(training_data)} samples total.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
