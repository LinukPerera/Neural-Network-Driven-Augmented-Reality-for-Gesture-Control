import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow import keras

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = keras.models.load_model("gesture_classification_model.h5")

# Gesture classes
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]
gesture_to_label = {gesture: idx for idx, gesture in enumerate(gesture_classes)}

# Define weights for velocity computation
WEIGHTS = {"wrist": 1.0, "fingertip": 6.0}
FINGERTIP_IDS = [8, 12, 16, 20]

# Start video capture
cap = cv2.VideoCapture(1)

def compute_angle(velocity):
    """ Compute movement angle in degrees from velocity vector. """
    vx, vy = velocity
    angle = np.arctan2(vy, vx) * 180 / np.pi
    if angle < 0:
        angle += 360
    return angle

def countdown_timer(message, duration=3):
    for i in range(duration, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"{message} {i}", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("Gesture Recording", frame_copy)
        cv2.waitKey(1000)

def record_gesture_sequence(gesture_name, samples_needed=10, sequence_duration=3):
    """ Record gesture sequences including recoils with swapped velocity thresholds for Swipe Left/Right. """
    data = []
    print(f"\nRecording {gesture_name} sequences ({samples_needed} samples needed)")
    
    for sample_count in range(samples_needed):
        print(f"\nGet Ready for {gesture_name} Sequence {sample_count + 1}...")
        countdown_timer("Get Ready", 3)

        print("Recording sequence (perform gesture and natural recoil)...")
        sequence_data = []
        prev_positions = {}
        prev_time = time.time()
        start_time = prev_time
        prev_direction = None
        prev_velocity_norm = 0

        while time.time() - start_time < sequence_duration:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            current_time = time.time()

            # Initialize gesture as "Rest" by default
            gesture = "Rest"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    h, w, _ = frame.shape

                    velocities_list = []
                    total_weight = 0

                    for idx in [0] + FINGERTIP_IDS:
                        landmark = hand_landmarks.landmark[idx]
                        curr_pos = np.array([landmark.x * w, landmark.y * h])
                        weight = WEIGHTS["wrist"] if idx == 0 else WEIGHTS["fingertip"]

                        if idx in prev_positions and prev_time is not None:
                            delta_pos = curr_pos - prev_positions[idx]
                            delta_time = current_time - prev_time
                            if delta_time > 0:  # Avoid division by zero
                                velocity = (delta_pos / delta_time) * weight
                                velocities_list.append(velocity)
                                total_weight += weight

                        prev_positions[idx] = curr_pos

                    prev_time = current_time

                    if velocities_list:
                        weighted_velocity = np.sum(velocities_list, axis=0) / total_weight
                        velocity_x = weighted_velocity[0]  # Raw x velocity
                        abs_velocity = np.abs(weighted_velocity)
                        direction = np.sign(weighted_velocity)
                        angle = compute_angle(weighted_velocity)
                        velocity_norm = np.linalg.norm(abs_velocity)

                        # Recoil detection
                        is_recoil = False
                        if prev_direction is not None and velocity_norm < 300 and velocity_norm < 0.5 * prev_velocity_norm:
                            if (gesture_name == "Swipe Left" and direction[0] == -1.0) or \
                               (gesture_name == "Swipe Right" and direction[0] == 1.0):
                                is_recoil = True

                        # Swapped velocity thresholds for Swipe Left and Swipe Right
                        gesture_idx = -1  # Default to "Rest"
                        if gesture_name == "Swipe Left" and not is_recoil:
                            if velocity_x > -1000:  # Strong rightward motion → Swipe Left
                                gesture = "Swipe Left"
                                gesture_idx = gesture_to_label["Swipe Left"]
                            else:
                                gesture = "Rest"
                        elif gesture_name == "Swipe Right" and not is_recoil:
                            if velocity_x < 1000:  # Strong leftward motion → Swipe Right
                                gesture = "Swipe Right"
                                gesture_idx = gesture_to_label["Swipe Right"]
                            else:
                                gesture = "Rest"
                        else:
                            # Use model prediction for other gestures
                            input_velocity = np.pad(weighted_velocity, (0, 3), 'constant', constant_values=0).reshape(1, -1)
                            prediction = model.predict(input_velocity, verbose=0)
                            confidence = np.max(prediction)
                            gesture_idx = np.argmax(prediction)
                            if confidence > 0.9 and not is_recoil:
                                gesture = gesture_classes[gesture_idx]
                            else:
                                gesture = "Rest"
                                gesture_idx = -1

                        # Store features if not recoil
                        if gesture != "Rest":
                            sample = np.array([abs_velocity[0], abs_velocity[1], direction[0], direction[1], angle, gesture_idx])
                            sequence_data.append(sample)

                        prev_direction = direction[0]
                        prev_velocity_norm = velocity_norm

            cv2.putText(frame, f"Recording {gesture_name}: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Gesture Recording", frame)
            cv2.waitKey(1)

        print(f"Sequence {sample_count + 1} recorded.")
        if sequence_data:
            data.extend(sequence_data)
        time.sleep(2)

    return data

# Collect data
unlabeled_data = []
samples_per_gesture = 15

print("Welcome to Gesture and Recoil Recording!")
print("Perform gestures naturally, including recoils, for 3 seconds per sample.")

for gesture in gesture_classes:
    print(f"\nStarting {gesture}...")
    gesture_data = record_gesture_sequence(gesture, samples_needed=samples_per_gesture)
    if gesture_data:
        unlabeled_data.extend(gesture_data)
        print(f"Collected {len(gesture_data)} frames for {gesture}.")
    else:
        print(f"Warning: No valid frames collected for {gesture}.")

# Save data
np.save("unlabeled_gesture_data.npy", np.array(unlabeled_data))
print(f"\nData saved! Recorded {len(unlabeled_data)} frames total.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
