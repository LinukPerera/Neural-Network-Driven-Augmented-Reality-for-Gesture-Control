#DataPreprocessor.py
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class GestureDataPreprocessor:
    def __init__(self, train_csv, val_test_csv, test_size=0.5):
        """
        train_csv: CSV file for training data.
        val_test_csv: CSV file for validation & testing data.
        test_size: Percentage of validation-test split (default 50%).
        """
        self.train_csv = train_csv
        self.val_test_csv = val_test_csv
        self.test_size = test_size

    def load_data(self, csv_file):
        """Loads gesture sequences grouped by sequence_id from a given CSV file."""
        sequences = {}
        labels = {}

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 67:  # Ensure correct row length
                    continue

                label = row[0]  
                sequence_id = row[1]  
                try:
                    frame_index = int(float(row[2]))  
                except ValueError:
                    continue

                try:
                    landmarks = list(map(float, row[4:]))  
                except ValueError:
                    continue

                if sequence_id not in sequences:
                    sequences[sequence_id] = []
                    labels[sequence_id] = label

                sequences[sequence_id].append((frame_index, landmarks))

        sorted_sequences = []
        sorted_labels = []
        for seq_id, frames in sequences.items():
            frames.sort(key=lambda x: x[0])  
            sorted_sequences.append(np.array([frame[1] for frame in frames]))  
            sorted_labels.append(labels[seq_id])

        return np.array(sorted_sequences), np.array(sorted_labels)

    def normalize_data(self, sequences):
        """Normalize landmark coordinates to [0, 1] range."""
        scaler = MinMaxScaler()
        sequences_reshaped = sequences.reshape(sequences.shape[0], -1)
        sequences_normalized = scaler.fit_transform(sequences_reshaped)
        return sequences_normalized.reshape(sequences.shape)  

    def encode_labels(self, train_labels, val_test_labels):
        """Ensures label encoding is consistent across both datasets."""
        label_encoder = LabelEncoder()
        all_labels = np.concatenate((train_labels, val_test_labels))
        label_encoder.fit(all_labels)  # Fit on both datasets to ensure consistent encoding

        return (
            label_encoder.transform(train_labels),
            label_encoder.transform(val_test_labels),
            label_encoder
        )

    def split_val_test(self, X, y):
        """Splits validation & test data while maintaining label distribution."""
        X_val, X_test, y_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        return X_val, X_test, y_val, y_test

    def preprocess(self):
        """Full preprocessing pipeline."""
        # Load training data
        X_train, y_train = self.load_data(self.train_csv)

        # Load validation + testing data
        X_val_test, y_val_test = self.load_data(self.val_test_csv)

        # Normalize data
        X_train = self.normalize_data(X_train)
        X_val_test = self.normalize_data(X_val_test)

        # Encode labels
        y_train, y_val_test, label_encoder = self.encode_labels(y_train, y_val_test)

        # Split validation and test sets (Taken from the second file to further reduce Overfit)
        X_val, X_test, y_val, y_test = self.split_val_test(X_val_test, y_val_test)

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder
        }


if __name__ == "__main__":
    preprocessor = GestureDataPreprocessor(
        train_csv='shuffeled.csv', 
        val_test_csv='shuffeled1.csv'
    )
    
    preprocessed_data = preprocessor.preprocess()

    # Save preprocessed data
    np.save('X_train.npy', preprocessed_data['X_train'])
    np.save('X_val.npy', preprocessed_data['X_val'])
    np.save('X_test.npy', preprocessed_data['X_test'])
    np.save('y_train.npy', preprocessed_data['y_train'])
    np.save('y_val.npy', preprocessed_data['y_val'])
    np.save('y_test.npy', preprocessed_data['y_test'])

    print("Preprocessing complete! Data saved to .npy files.")
