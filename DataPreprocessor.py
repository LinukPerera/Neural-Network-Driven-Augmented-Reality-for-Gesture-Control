#DataPreprocessor.py
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# In DataPreprocessor.py, add this at the end of the preprocess method:


class GestureDataPreprocessor:
    def __init__(self, input_csv, sequence_length=15, test_size=0.2, val_size=0.1):
        self.input_csv = input_csv
        self.sequence_length = sequence_length  # Number of frames per sequence (set to 15)
        self.test_size = test_size
        self.val_size = val_size

    def load_data(self):
        """Load data from CSV file."""
        labels = []
        sequences = []

        with open(self.input_csv, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            for row in reader:
                # Check if the row has the correct number of columns
                if len(row) != 65:  # 1 (label) + 1 (timestamp) + 63 (landmarks)
                    print(f"Skipping row with incorrect length: {len(row)}")
                    continue

                label = row[0]
                landmarks = list(map(float, row[2:]))  # Skip timestamp
                sequences.append(landmarks)
                labels.append(label)

        # Convert to numpy arrays
        sequences = np.array(sequences)
        labels = np.array(labels)

        # Check shapes
        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")

        return sequences, labels

    def normalize_data(self, sequences):
        """Normalize landmark coordinates to [0, 1] range."""
        scaler = MinMaxScaler()
        sequences_normalized = scaler.fit_transform(sequences)
        return sequences_normalized

    def create_sequences(self, sequences, labels):
        """Create sequences of frames for dynamic gestures."""
        X, y = [], []
        for i in range(len(sequences) - self.sequence_length):
            sequence = sequences[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]  # Use the last label in the sequence
            X.append(sequence)
            y.append(label)
        return np.array(X), np.array(y)

    def encode_labels(self, labels):
        """Encode string labels into integers."""
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        return labels_encoded, label_encoder

    def split_data(self, X, y):
        """Split data into training, validation, and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self):
        """Run the entire preprocessing pipeline."""
        # Step 1: Load data
        sequences, labels = self.load_data()

        # Step 2: Normalize data
        sequences_normalized = self.normalize_data(sequences)

        # Step 3: Create sequences
        X, y = self.create_sequences(sequences_normalized, labels)

        # Step 4: Encode labels
        y_encoded, label_encoder = self.encode_labels(y)

        # Step 5: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y_encoded)

        # Return preprocessed data
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder
        }

# Example usage
if __name__ == "__main__":
    preprocessor = GestureDataPreprocessor(input_csv='augmented_gesture_data.csv', sequence_length=15)
    preprocessed_data = preprocessor.preprocess()

    # Save preprocessed data to files
    np.save('X_train.npy', preprocessed_data['X_train'])
    np.save('X_val.npy', preprocessed_data['X_val'])
    np.save('X_test.npy', preprocessed_data['X_test'])
    np.save('y_train.npy', preprocessed_data['y_train'])
    np.save('y_val.npy', preprocessed_data['y_val'])
    np.save('y_test.npy', preprocessed_data['y_test'])

    print("Preprocessing complete! Data saved to .npy files.")
