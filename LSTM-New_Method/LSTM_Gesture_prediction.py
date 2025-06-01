import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
import logging
from collections import deque
import pandas as pd
import warnings

# Suppress mediapipe protobuf deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === CONFIGURATION ===
MODEL_PATH = 'outputs/gesture_lstm_model.h5'
SCALER_PATH = 'outputs/scaler.pkl'
ENCODER_PATH = 'outputs/label_encoder.pkl'
PREDICTIONS_LOG = 'predictions.csv'

WINDOW_SIZE = 15
FEATURES = 3
CONFIDENCE_THRESHOLD = 0.90
VELOCITY_THRESHOLD = 200

WEIGHTS = {"wrist": 1.0, "fingertip": 7.0}
FINGERTIP_IDS = [8, 12, 16, 20]

# === INITIATIONS ===
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=WINDOW_SIZE)

# UI State
last_prediction = "Waiting..."
prediction_time = 0
prev_positions = {}
prev_time = None
predictions = []

def extract_features(frame, results, prev_positions, prev_time, current_time):
    if not results.multi_hand_landmarks:
        return None, prev_positions, prev_time, np.array([0.0, 0.0])

    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = frame.shape

    velocities = []
    total_weight = 0
    curr_positions = {}

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

        curr_positions[idx] = curr_pos

    weighted_velocity = np.sum(velocities, axis=0) / total_weight if velocities and total_weight > 0 else np.array([0.0, 0.0])

    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[8]
    dx = middle_tip.x * w - wrist.x * w
    dy = middle_tip.y * h - wrist.y * h
    angle = np.arctan2(dy, dx)

    return np.array([*weighted_velocity, angle]), curr_positions, current_time, weighted_velocity

def draw_prediction_overlay(frame, prediction):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

def run_gesture_ui():
    global last_prediction, prediction_time, prev_positions, prev_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        current_time = time.time()

        features, prev_positions, prev_time, velocity_vector = extract_features(
            frame, results, prev_positions, prev_time, current_time)

        if features is not None:
            frame_buffer.append(features)

        if len(frame_buffer) == WINDOW_SIZE:
            velocity_norm = np.linalg.norm(velocity_vector)
            if velocity_norm < VELOCITY_THRESHOLD:
                gesture = "Rest"
                confidence = 1.0
            else:
                sequence = np.array(frame_buffer)
                scaled = scaler.transform(sequence.reshape(-1, FEATURES)).reshape(1, WINDOW_SIZE, FEATURES)
                prediction = model.predict(scaled, verbose=0)
                confidence = np.max(prediction)
                if confidence < CONFIDENCE_THRESHOLD:
                    gesture = "Rest"
                else:
                    gesture = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            if gesture != last_prediction:
                last_prediction = gesture
                prediction_time = current_time
                predictions.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'gesture': gesture,
                    'confidence': round(confidence, 3)
                })
                logging.info(f"Gesture: {gesture} ({confidence:.2f})")

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if current_time - prediction_time < 0.8:
            draw_prediction_overlay(frame, last_prediction)

        cv2.imshow("LSTM Gesture Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save predictions
    if predictions:
        pd.DataFrame(predictions).to_csv(PREDICTIONS_LOG, index=False)
        logging.info(f"Saved predictions to {PREDICTIONS_LOG}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    run_gesture_ui()