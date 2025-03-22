#GestureCollection.py

#GestureCollection.py
import csv
import cv2
import mediapipe as mp
import time
from collections import deque

# Configuration
COLLECTION_WINDOW = 1.0  # 1 second per gesture capture
ADJUSTMENT_WINDOW = 1.0   # 1 second rest period
MIN_HAND_CONFIDENCE = 0.7
NUM_SAMPLES = 100  # Number of times to repeat the same gesture
FRAMES_PER_SAMPLE = 15  # Fixed number of frames per gesture

class GestureCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=MIN_HAND_CONFIDENCE,
            min_tracking_confidence=MIN_HAND_CONFIDENCE
        )
        self.cap = cv2.VideoCapture(0)
        self.writer = None
        self.csv_file = None
        self.current_gesture = None
        self.last_sample_time = 0
        self.state = "WAITING"
        self.buffer = deque(maxlen=FRAMES_PER_SAMPLE)  # Fixed-size buffer
        self.num_samples = NUM_SAMPLES
        self.current_sample_count = 0
        self.frame_counter = 0  # Track frames during recording

    def initialize_csv(self):
        self.csv_file = open(f'gesture_data1.csv', mode='a', newline='')
        self.writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            header = ['label', 'timestamp'] + \
                     [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
            self.writer.writerow(header)

    def collect_samples(self):
        self.initialize_csv()

        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break

            current_time = time.time()
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            self.handle_key_input()
            self.update_state(current_time)
            self.process_frame(results, current_time, img)

            cv2.imshow("Gesture Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def handle_key_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and self.state == "WAITING":
            self.current_gesture = input("Enter gesture label: ")
            self.current_sample_count = 0
            self.state = "GO"
            self.last_sample_time = time.time()
            print(f"Starting gesture collection for {self.current_gesture} ({self.num_samples} times)...")

    def update_state(self, current_time):
        if self.state == "GO":
            if current_time - self.last_sample_time >= 2.0:
                self.state = "RECORDING"
                self.last_sample_time = current_time
                self.frame_counter = 0  # Reset frame counter
                print(f"Perform gesture NOW: {self.current_gesture}")

        elif self.state == "RECORDING":
            if self.frame_counter >= FRAMES_PER_SAMPLE:
                self.state = "RECORDING_COMPLETE"
                self.last_sample_time = current_time
                print(f"Finished recording {FRAMES_PER_SAMPLE} frames for {self.current_gesture}.")

        elif self.state == "RECORDING_COMPLETE":
            self.save_buffer_to_csv()
            self.current_sample_count += 1
            if self.current_sample_count < self.num_samples:
                self.state = "WAIT"
                self.last_sample_time = current_time
                print(f"Saved sample {self.current_sample_count}/{self.num_samples}. Get ready...")
            else:
                self.state = "WAITING"
                print(f"Collected {self.num_samples} samples for {self.current_gesture}. Press 's' for a new gesture.")

        elif self.state == "WAIT":
            if current_time - self.last_sample_time >= ADJUSTMENT_WINDOW:
                self.state = "GO"
                self.last_sample_time = current_time
                print("Prepare to perform the next repetition...")

    def process_frame(self, results, timestamp, img):
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                        [lm.y for lm in hand_landmarks.landmark] + \
                        [lm.z for lm in hand_landmarks.landmark]

            if self.state == "RECORDING":
                self.buffer.append((timestamp, landmarks))
                self.frame_counter += 1  # Increment frame counter

            self.draw_landmarks(img, hand_landmarks)

        self.draw_ui(img)

    def draw_landmarks(self, img, landmarks):
        mp.solutions.drawing_utils.draw_landmarks(
            img, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def draw_ui(self, img):
        height, width, _ = img.shape

        # Grid settings (3x3 grid)
        num_lines = 2  # Number of lines per axis (creating 3 sections)

        # Draw vertical lines
        for i in range(1, num_lines + 1):
            x = int(i * width / (num_lines + 1))
            cv2.line(img, (x, 0), (x, height), (100, 100, 100), 1)

        # Draw horizontal lines
        for i in range(1, num_lines + 1):
            y = int(i * height / (num_lines + 1))
            cv2.line(img, (0, y), (width, y), (100, 100, 100), 1)

        # Draw status text
        status_text = {
            "WAITING": "Press 's' to start recording",
            "GO": "Get Ready! Perform the Gesture NOW!",
            "RECORDING": f"Recording ({self.frame_counter}/{FRAMES_PER_SAMPLE})...",
            "RECORDING_COMPLETE": "Recording Complete!",
            "WAIT": "Wait..."
        }

        color_map = {
            "WAITING": (0, 255, 0),
            "GO": (0, 0, 255),
            "RECORDING": (0, 255, 255),
            "RECORDING_COMPLETE": (255, 0, 0),
            "WAIT": (255, 165, 0)
        }

        cv2.putText(img, status_text[self.state], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map[self.state], 2)
        cv2.putText(img, f"Current Gesture: {self.current_gesture or 'None'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def save_buffer_to_csv(self):
        for timestamp, landmarks in self.buffer:
            row = [self.current_gesture, timestamp] + landmarks
            self.writer.writerow(row)
        self.csv_file.flush()
        self.buffer.clear()
        print(f"Saved {len(self.buffer)} samples for {self.current_gesture}")

    def cleanup(self):
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = GestureCollector()
    collector.collect_samples()
