import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

class GestureRecognizer:
    def __init__(self, model_path, input_csv):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.model = tf.keras.models.load_model(model_path)
        self.sequence = []
        self.sequence_length = 15
        self.gesture_history = []  # Store past gestures for smoothing
        
        # Extract unique classes from the original CSV
        self.classes = self._get_unique_classes(input_csv)

    def _get_unique_classes(self, input_csv):
        import csv
        classes = set()
        with open(input_csv, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                classes.add(row[0])
        return sorted(list(classes))

    def preprocess_landmarks(self, landmarks):
        # Convert landmarks to flat list and normalize
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return coords / np.max(coords)

    def recognize_gesture(self):
        if len(self.sequence) == self.sequence_length:
            input_sequence = np.array(self.sequence).reshape(1, self.sequence_length, -1)
            prediction = self.model.predict(input_sequence)
            confidence = np.max(prediction)

            # Gesture smoothing with history (most common gesture)
            self.gesture_history.append(np.argmax(prediction))
            if len(self.gesture_history) > 5:  # Adjust N for smoother results
                self.gesture_history.pop(0)
            
            # Only return gesture if confidence is high enough
            if confidence > 0.8:  # Adjust confidence threshold as needed
                gesture_index = np.argmax(prediction)
                return self.classes[gesture_index]
            else:
                return "Uncertain"  # Or return None or a placeholder gesture
        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                processed_landmarks = self.preprocess_landmarks(landmarks)
                
                self.sequence.append(processed_landmarks)
                if len(self.sequence) > self.sequence_length:
                    self.sequence.pop(0)
                
                gesture = self.recognize_gesture()
                if gesture:
                    cv2.putText(frame, f'Gesture: {gesture}', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
recognizer = GestureRecognizer('gesture_lstm_model.h5', 'shuffeled.csv')
recognizer.run()