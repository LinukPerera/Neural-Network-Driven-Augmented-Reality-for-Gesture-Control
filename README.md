# Neural-Network-Driven-Augmented-Reality-for-Gesture-Control
Github Repo of my final year project under the topic Neural Network-Driven Augmented Reality for Gesture Control
Here’s a detailed and informative `README.md` template that you can use for your GitHub project. This file will describe your project, its components, usage, and how to run each script.

## Gesture Recognition Using LSTM (Long Short-Term Memory) Networks

This project aims to develop a gesture recognition system using LSTM networks. The project captures hand gesture data, performs data augmentation, preprocesses the data, trains an LSTM model, and evaluates its performance. The goal is to recognize dynamic gestures captured via a camera and classify them.

## Project Overview

The project consists of several Python scripts that work together to:

1. **Collect gesture data** using a camera.
2. **Augment** the collected data to increase the dataset's size and diversity.
3. **Preprocess** the data for training, including normalizing and creating sequences.
4. **Train an LSTM model** for gesture recognition.
5. **Evaluate the model** and visualize the results, including a classification report and confusion matrix.

The dataset is collected at 15 frames per sample, with 100 samples per gesture. Data augmentation multiplies the dataset by 25 times by applying different transformations (shifting, scaling, etc.).

## Project Components

The project contains the following scripts:

1. **`GestureCollection.py`**: 
    - Captures gesture data at 15 frames per sample.
    - Collects 100 samples for each gesture.
    - Includes a user-friendly guide to start and stop gesture collection.
    - Data is saved in CSV format with landmarks and corresponding gesture labels.

2. **`DataAugmentation.py`**: 
    - Increases the dataset size by applying augmentation techniques.
    - Uses shifting (top left, top middle, top right, left middle, right middle, bottom left and right) and scaling for Z-axis transformation.

3. **`DataPreprocessor.py`**: 
    - Loads and preprocesses the collected and augmented gesture data.
    - Normalizes the landmarks, creates sequences for gesture frames, encodes labels, and splits the data into training, validation, and test sets.

4. **`ModelEvaluation.py`**: 
    - Loads the trained LSTM model and evaluates its performance.
    - Generates a classification report and confusion matrix.
    - Visualizes the confusion matrix using a heatmap.

5. **`LSTMGestureClassificationModel.py`**: 
    - Defines and trains the LSTM model for gesture recognition.
    - Includes model architecture, loss function, optimizer, and training steps.
  
## Installation

### Prerequisites

To run the scripts in this project, you need to have the following dependencies installed:

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (for model training and evaluation)

You can install the required packages using `pip`:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## Usage

### 1. Gesture Data Collection

Run **`GestureCollection.py`** to collect gesture data. You will be prompted to enter a label for each gesture.

```bash
python GestureCollection.py
```

- Press `'s'` to start collecting gestures.
- You will be asked to input a gesture label.
- The script will collect 100 samples per gesture and save them in a CSV file.

### 2. Data Augmentation

Once you've collected enough gesture data, you can use **`DataAugmentation.py`** to augment the data by applying transformations like shifting and scaling.

```bash
python DataAugmentation.py
```

This will generate augmented versions of the dataset and save them to new files.

### 3. Data Preprocessing

Run **`DataPreprocessor.py`** to preprocess the collected and augmented data. This step normalizes the data, creates sequences, and splits it into training, validation, and test sets.

```bash
python DataPreprocessor.py
```

This will preprocess the data and save the resulting arrays in `.npy` format, which will be used to train the model.

### 4. Train the LSTM Model

Use **`LSTMGestureClassificationModel.py`** to define, compile, and train the LSTM model.

```bash
python LSTMGestureClassificationModel.py
```

This script will train the model on the preprocessed data and save the trained model to a file (e.g., `gesture_lstm_model.h5`).

### 5. Model Evaluation

After training the model, use **`ModelEvaluation.py`** to load the trained model and evaluate its performance on the test set.

```bash
python ModelEvaluation.py
```

This will generate:
- A classification report.
- A confusion matrix heatmap showing the performance of the model.

### Example Files

- **Gesture Data (CSV)**: Data captured from **`GestureCollection.py`**.
- **Augmented Data (CSV)**: Data after applying transformations from **`DataAugmentation.py`**.
- **Preprocessed Data (Numpy `.npy` files)**: Data ready for training.
- **Model**: The trained LSTM model (`gesture_lstm_model.h5`).
- **Evaluation Results**: Classification report and confusion matrix heatmap.

## File Structure

```
.
├── GestureCollection.py              # Script to collect gesture data
├── DataAugmentation.py               # Script for data augmentation
├── DataPreprocessor.py               # Script to preprocess the data
├── ModelEvaluation.py                # Script to evaluate the model
├── LSTMGestureClassificationModel.py # Script to define and train the LSTM model
├── augmented_gesture_data.csv        # Example of collected and augmented data
├── gesture_lstm_model.h5            # Trained LSTM model (after training)
├── label_encoder.npy                # Label encoder (for evaluation)
└── confusion_matrix.png             # Confusion matrix heatmap (for evaluation)
```

## Model Performance

After training and evaluating the model, you'll have access to:
- **Classification report**: A detailed report of precision, recall, f1-score, and support for each gesture class.
- **Confusion matrix heatmap**: A visual representation of the model’s confusion matrix, showing true vs predicted labels.

## Contributing

If you have suggestions or improvements for this project, feel free to open an issue or submit a pull request. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
