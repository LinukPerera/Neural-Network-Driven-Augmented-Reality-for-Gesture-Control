import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_DIR = '/Users/linukperera/Programming/Python/Lv 06 Project/May - 5/gesture_data_trimmed'
GESTURE_CLASSES = ['Swipe Up', 'Swipe Down', 'Swipe Right', 'Swipe Left']
MAX_SEQ_LENGTH = 15  # Pad sequences to this length
OUTPUT_DIR = '/Users/linukperera/Programming/Python/Lv 06 Project/May - 5/outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'gesture_lstm_model.h5')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to encode gesture labels
label_encoder = LabelEncoder()
label_encoder.fit(GESTURE_CLASSES)
gesture_to_index = {gesture: i for i, gesture in enumerate(GESTURE_CLASSES)}

# Load all gesture sequences
X, y = [], []
gesture_counts = Counter()
sequence_lengths = []

logging.info(f"Loading data from {DATA_DIR}")
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                logging.warning(f"Empty CSV: {filename}")
                continue

            # Collect stats
            gesture_label = df['gesture'].iloc[0]
            if gesture_label not in GESTURE_CLASSES:
                logging.warning(f"Unknown gesture {gesture_label} in {filename}")
                continue
            gesture_counts[gesture_label] += 1
            sequence_lengths.append(len(df))

            # Drop the frame column and gesture label
            features = df[['x_vel', 'y_vel', 'angle']].values

            # Pad or truncate
            if len(features) < MAX_SEQ_LENGTH:
                pad_width = MAX_SEQ_LENGTH - len(features)
                features = np.pad(features, ((0, pad_width), (0, 0)), 'constant')
            else:
                features = features[:MAX_SEQ_LENGTH]

            X.append(features)
            y.append(gesture_to_index[gesture_label])
        except Exception as e:
            logging.error(f"Failed to process {filename}: {str(e)}")
            continue

if not X:
    logging.error("No valid data loaded. Check CSV files.")
    raise ValueError("No valid data loaded")

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(GESTURE_CLASSES))
logging.info(f"Loaded {len(X)} sequences. Gesture counts: {dict(gesture_counts)}")

# Standardize features
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])  # Flatten for scaling
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)  # Reshape back to (samples, MAX_SEQ_LENGTH, 3)
logging.info("Scaled features using StandardScaler")

# Save scaler and label encoder
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
logging.info(f"Saved scaler to {SCALER_PATH} and label encoder to {LABEL_ENCODER_PATH}")

# Visualize dataset distribution
plt.figure(figsize=(8, 5))
plt.bar(gesture_counts.keys(), gesture_counts.values(), color='skyblue')
plt.title("Gesture Class Distribution")
plt.xlabel("Gesture")
plt.ylabel("Number of Samples")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gesture_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(sequence_lengths, bins=20, color='salmon', edgecolor='black')
plt.title("Sequence Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sequence_length_distribution.png"))
plt.close()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
logging.info(f"Split data: {len(X_train)} training, {len(X_test)} testing samples")

# Build LSTM Model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(MAX_SEQ_LENGTH, 3)))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(GESTURE_CLASSES), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
logging.info("Built and compiled LSTM model")

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=30, batch_size=8, verbose=1)
logging.info("Completed model training")

# Save the model
model.save(MODEL_PATH)
logging.info(f"Saved model to {MODEL_PATH}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
plt.close()

# Evaluate model
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GESTURE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Print classification report to console
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=GESTURE_CLASSES))

print("\nModel and plots saved in:", os.path.abspath(OUTPUT_DIR))