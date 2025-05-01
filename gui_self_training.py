import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras

# Load the semi-supervised model (default to Mean Teacher; can switch to self_training_recoil_model.h5)
MODEL_PATH = "self_training_recoil_model.h5"  # Or "self_training_recoil_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Gesture labels (matches semi-supervised models)
gesture_classes = ["Swipe Up", "Swipe Down", "Swipe Right", "Swipe Left"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(1)

# Define weights for velocity contribution
WEIGHTS = {"wrist": 1.0, "fingertip": 7.0}

# Fingertip landmarks in Mediapipe (index, middle, ring, pinky)
FINGERTIP_IDS = [8, 12, 16, 20]

# UI Effect Variables
last_prediction = "Waiting..."
prediction_time = 0
prev_velocity_norm = 0
prev_x_direction = 0

def compute_features(weighted_velocity):
    """ Compute features for model input: abs velocities, directions, and angle. """
    velocity_x, velocity_y = weighted_velocity
    abs_x_velocity = abs(velocity_x)
    abs_y_velocity = abs(velocity_y)
    x_direction = np.sign(velocity_x)
    y_direction = np.sign(velocity_y)
    
    # Compute movement angle in degrees
    angle = np.arctan2(velocity_y, velocity_x) * 180 / np.pi
    if angle < 0:
        angle += 360
    
    return np.array([abs_x_velocity, abs_y_velocity, x_direction, y_direction, angle])

def normalize_features(features):
    """ Normalize features to match training data. """
    features = features.copy()
    features[0] = features[0] / 1000  # abs_x_velocity
    features[1] = features[1] / 1000  # abs_y_velocity
    features[2] = (features[2] + 1) / 2  # x_direction: [-1, 1] to [0, 1]
    features[3] = (features[3] + 1) / 2  # y_direction: [-1, 1] to [0, 1]
    features[4] = features[4] / 360  # movement_angle_degrees
    return features

def classify_gesture(weighted_velocity, prev_velocity_norm, prev_x_direction):
    """ Classifies gesture with manual thresholds, recoil detection, and model prediction. """
    velocity_x = weighted_velocity[0]
    velocity_norm = np.linalg.norm(weighted_velocity)
    
    # Rest detection: low velocity
    if velocity_norm < 500:
        return "Rest"
    
    # Recoil detection: low velocity, opposite direction
    curr_x_direction = np.sign(velocity_x)
    if velocity_norm < 300 and velocity_norm < 0.5 * prev_velocity_norm and curr_x_direction != prev_x_direction:
        return "Rest"
    
    # Manual classification for Left/Right (swapped)
    if velocity_x < -1000:  # Strong leftward motion → Swipe Right
        return "Swipe Right"
    elif velocity_x > 1000:  # Strong rightward motion → Swipe Left
        return "Swipe Left"
    
    # Compute and normalize features for model
    features = compute_features(weighted_velocity)
    normalized_features = normalize_features(features)
    normalized_features = normalized_features.reshape(1, -1)
    
    # Model prediction
    prediction = model.predict(normalized_features, verbose=0)
    max_confidence = np.max(prediction)
    
    if max_confidence < 0.90:
        return "Rest"
    
    return gesture_classes[np.argmax(prediction)]

def draw_prediction_overlay(frame, prediction):
    """ Darkens the background and displays the prediction in the center of the screen. """
    overlay = frame.copy()
    
    # Darken the screen slightly
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    alpha = 0.5  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Display prediction text
    text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

def predict_gesture():
    """ Continuously detects hand movement and predicts gesture instantly. """
    global last_prediction, prediction_time, prev_velocity_norm, prev_x_direction

    prev_positions = {}
    prev_time = None

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

                velocities = []
                total_weight = 0

                for idx in [0] + FINGERTIP_IDS:
                    landmark = hand_landmarks.landmark[idx]
                    curr_pos = np.array([landmark.x * w, landmark.y * h])
                    weight = WEIGHTS["wrist"] if idx == 0 else WEIGHTS["fingertip"]

                    if idx in prev_positions and prev_time is not None:
                        delta_pos = curr_pos - prev_positions[idx]
                        delta_time = current_time - prev_time
                        if delta_time > 0:
                            velocity = (delta_pos / delta_time) * weight
                            velocities.append(velocity)
                            total_weight += weight

                    prev_positions[idx] = curr_pos

                prev_time = current_time

                # Compute weighted velocity
                if velocities and total_weight > 0:
                    weighted_velocity = np.sum(velocities, axis=0) / total_weight
                    new_prediction = classify_gesture(weighted_velocity, prev_velocity_norm, prev_x_direction)
                    
                    # Update previous state for recoil detection
                    prev_velocity_norm = np.linalg.norm(weighted_velocity)
                    prev_x_direction = np.sign(weighted_velocity[0])
                    
                    if new_prediction != last_prediction:
                        last_prediction = new_prediction
                        prediction_time = time.time()

        # Display modern UI
        if time.time() - prediction_time < 0.8:  # Darken background for 0.8 seconds
            draw_prediction_overlay(frame, last_prediction)

        cv2.imshow("Live Gesture Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

predict_gesture()