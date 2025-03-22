import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load model
model = tf.keras.models.load_model('gesture_lstm_model.h5')

# Get unique classes from the test labels
unique_classes = np.unique(y_test)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_classes, 
    target_names=[str(cls) for cls in unique_classes]))

# Confusion matrix visualization
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', 
    xticklabels=[str(cls) for cls in unique_classes],
    yticklabels=[str(cls) for cls in unique_classes])
plt.title('Gesture Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')


import numpy as np
from sklearn.preprocessing import LabelEncoder
# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
#Check Dataset Shape

print("X_train shape:", X_train.shape)  # Expected: (num_samples, 15, 63)
print("y_train shape:", y_train.shape)  # Expected: (num_samples,)
print("Unique labels in y_train:", np.unique(y_train))

#✅ X_train.shape[0] > 1 (You must have multiple samples)
#✅ y_train.shape[0] == X_train.shape[0] (Matching labels & samples)
#✅ np.unique(y_train) has more than 1 value (Avoid single-class issue)
#🔴 If X_train shape is (1, 15, 63) → Your dataset is too small
#🔴 If np.unique(y_train) is only [0] → Labels are not encoded properly

#Check If Data Is All Zeros

print("X_train min:", np.min(X_train)) #✅ X_train min and max should not be 0.0
print("X_train max:", np.max(X_train))
print("X_train mean:", np.mean(X_train)) #✅ X_train mean should be some non-zero value

#Check If Labels Are Encoded Correctly

print("y_train dtype:", y_train.dtype) 
print("y_train unique values:", np.unique(y_train))

#✅ y_train should be of type int
#✅ Unique values should be like [0, 1, 2, 3, ...] (not floats or strings)
#🔴 If dtype is float → Your labels might be one-hot encoded (fix needed)
#🔴 If only [0] appears → Label encoding might be incorrect

#Check If Data Splitting Is Wrong

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

#✅ Both X_train and X_val should have non-zero samples
#🔴 If X_train.shape[0] == 0 → Your dataset is too small for splitting

#Check If Model is Seeing Any Data

print("First training sample:")
print("X_train[0]:", X_train[0])
print("y_train[0]:", y_train[0])

#✅ X_train[0] should have non-zero values
#✅ y_train[0] should be a valid label (integer)

#Check Data Labels
print("y_train raw values:", y_train)

#Fix Label Encoding

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

print("New unique labels in y_train:", np.unique(y_train))

#Check Data Labels
print("y_train raw values:", y_train)

print(f"Unique labels in y_train: {set(y_train)}")
print(f"Unique labels in y_val: {set(y_val)}")


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Load CSV file
df = pd.read_csv("augmented_gesture_data.csv")

# Count occurrences of each label
print(df["label"].value_counts())

# Load features and labels
X = df.iloc[:, 3:].values  # Features
y = df["label"].values  # Encoded labels

# Stratified split ensures both classes appear in train & validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check again
print("Unique labels in y_train:", set(y_train))
print("Unique labels in y_val:", set(y_val))

# Load CSV file
df = pd.read_csv("augmented_gesture_data.csv")

# Print unique labels before encoding
print("Unique labels from CSV:", df["label"].unique())


# Load CSV
df = pd.read_csv("augmented_gesture_data.csv")


# Encode labels correctly
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Print unique encoded labels
print("Encoded labels:", df["label"].unique())

# Save as NumPy arrays
X = df.iloc[:, 3:].values  # All feature columns
y = df["label"].values  # Encoded labels

print("Unique labels in y:", np.unique(y))

# Load CSV
df = pd.read_csv("augmented_gesture_data.csv")

# 🔧 Normalize labels (fix inconsistent naming)
df["label"] = df["label"].replace({
    "swipe_left": "left_swipe"  # Standardize both to 'left_swipe'
})

# Encode labels correctly
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Print unique encoded labels
print("Final unique labels:", df["label"].unique())

# Save as NumPy arrays
X = df.iloc[:, 3:].values  # All feature columns
y = df["label"].values  # Encoded labels

print("Final unique labels in y:", y)

# Load CSV
df = pd.read_csv("augmented_gesture_data.csv")

# Encode labels correctly
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Print unique encoded labels
print("Encoded labels:", df["label"].unique())

# Save as NumPy arrays
X = df.iloc[:, 3:].values  # All feature columns
y = df["label"].values  # Encoded labels

print("Unique labels in y:", np.unique(y))
