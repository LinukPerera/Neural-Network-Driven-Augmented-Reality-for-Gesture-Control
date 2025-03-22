import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("gesture_classification_model.h5")

# Gesture labels (added "Rest")
gesture_classes = ["Rest", "Swipe Up", "Swipe Down", "Swipe Left", "Swipe Right"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Define weights for velocity contribution (fingertips get higher importance)
WEIGHTS = {
    "wrist": 1.0,
    "fingertip": 3.0
}

# Fingertip landmarks in Mediapipe (index, middle, ring, pinky)
FINGERTIP_IDS = [8, 12, 16, 20]

def classify_gesture(weighted_velocity):
    """ Classifies gesture based on weighted velocity data with confidence threshold. """
    # Check if the velocity magnitude is below the "Rest" threshold
    if np.linalg.norm(weighted_velocity) < 500:  # Threshold for "Rest"
        return "Rest"

    # Ensure the velocity vector is in the correct shape for the model (1, 5)
    if len(weighted_velocity) == 2:
        weighted_velocity = np.pad(weighted_velocity, (0, 3), 'constant', constant_values=0)

    weighted_velocity = weighted_velocity.reshape(1, -1)
    prediction = model.predict(weighted_velocity)
    max_confidence = np.max(prediction)  # Get the highest confidence score

    # If confidence is less than 99%, return "Rest"
    if max_confidence < 0.99:
        return "Rest"

    # Otherwise, return the predicted gesture (offset by 1 to skip "Rest")
    return gesture_classes[np.argmax(prediction) + 1]

def predict_gesture():
    """ Continuously detects hand movement and predicts gesture instantly. """
    prev_positions = {}
    prev_time = None
    last_prediction = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape

                # Calculate weighted velocity using fingertips and wrist
                velocities = []
                total_weight = 0

                for idx in [0] + FINGERTIP_IDS:  # Include wrist (id=0) and fingertips
                    landmark = hand_landmarks.landmark[idx]
                    curr_pos = np.array([landmark.x * w, landmark.y * h])

                    if idx == 0:
                        weight = WEIGHTS["wrist"]
                    else:
                        weight = WEIGHTS["fingertip"]

                    if idx in prev_positions and prev_time is not None:
                        delta_pos = curr_pos - prev_positions[idx]
                        delta_time = current_time - prev_time
                        velocity = (delta_pos / delta_time) * weight
                        velocities.append(velocity)
                        total_weight += weight

                    prev_positions[idx] = curr_pos

                prev_time = current_time

                # Compute weighted average velocity
                if velocities:
                    weighted_velocity = np.sum(velocities, axis=0) / total_weight
                    last_prediction = classify_gesture(weighted_velocity)
                    print(f"Predicted Gesture: {last_prediction}")

        # Display video feed with latest prediction
        cv2.putText(frame, f"Gesture: {last_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Gesture Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            cap.release()
            cv2.destroyAllWindows()
            return

predict_gesture()