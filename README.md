# Neural-Network-Driven-Augmented-Reality-for-Gesture-Control
Github Repo of my final year project under the topic Neural Network-Driven Augmented Reality for Gesture Control
Here’s a detailed and informative `README.md` template that you can use for your GitHub project. This file will describe your project, its components, usage, and how to run each script.

## There's 2 Proposed ML Methodologies

### Method 1 Gesture Recognition Using CNN (Convolutional Neural Network) and LSTM (Long Short-Term Memory) Networks

This approach aims to capture key points in a human hand to develop a gesture recognition system using HPE (Hand Pose Estimation) techniques employing a CNN (Convolutional Neural Network) and LSTM networks.

### Method 2 Gesture Recognition Using CNN (Convolutional Neural Network) and Dense NN (Neural Network)

This approach aims to capture weighted human hand velocity data using HPE (Hand Pose Estimation) techniques employing a CNN (Convolutional Neural Network) and use a Dense Neural Network for Classification of gestures. This Method prooved not only to be lighter but also more accurate and faster.


## Project Overview

Method 1 consists of several Python scripts that work together to:

1. **Collect gesture data** using a camera.
2. **Augment** the collected data to increase the dataset's size and diversity.
3. **Shuffle** the now Augmented data.
4. **Preprocess** the data for training, including normalizing and creating sequences.
5. **Train an LSTM model** for gesture recognition.
6. **Evaluate the model** and visualize the results, including a classification report and confusion matrix.

The dataset is collected at 15 frames per sample, with 100 samples per gesture. Data augmentation multiplies the dataset by 25 times by applying different transformations (shifting, scaling, etc.).

Method 2 consists of several Python scripts that work together to:

1. **Collect gesture data** using a camera.
2. **Evaluate the recorded data** and visualize the results.
3. **Preprocess** the data for training, including normalizing and creating sequences.
4. **Evaluate the Preprocessed data** and visualize the results.
5. **Train an LSTM model** for gesture recognition.
6. **Evaluate the model** and visualize the results, including a classification report and confusion matrix.

## Project Components

The method 1 contains the following scripts:

1. **`GestureCollection.py`**: 
    - Captures gesture data at 15 frames per sample.
    - Collects 100 samples for each gesture.
    - Includes a user-friendly guide to start and stop gesture collection.
    - Data is saved in CSV format with landmarks and corresponding gesture labels.

2. **`DataAugmentation.py`**: 
    - Increases the dataset size by applying augmentation techniques.
    - Uses shifting (top left, top middle, top right, left middle, right middle, bottom left and right) and scaling for Z-axis transformation.
  
3. **`DataShuffle.py`**: 
    - Shuffles each data sample, (Each sample has 15 Frames)
    - Shuffled data reduces the risk of overfitting

5. **`DataPreprocessor.py`**: 
    - Loads and preprocesses the collected and augmented gesture data.
    - Normalizes the landmarks, creates sequences for gesture frames, encodes labels, and splits the data into training, validation, and test sets.

6. **`ModelEvaluation.py`**: 
    - Loads the trained LSTM model and evaluates its performance.
    - Generates a classification report and confusion matrix.
    - Visualizes the confusion matrix using a heatmap.

7. **`LSTMGestureClassificationModel.py`**: 
    - Defines and trains the LSTM model for gesture recognition.
    - Includes model architecture, loss function, optimizer, and training steps.
  
8.  **`GestureRecognition.py`**
    - Apon running all prerequisite scripts, (1 through 7) a .h5 file will be created along side a csv file thats used for determining the gesture classes that the model can recognize


The method 2 contains the following scripts:

1. **`train.py`**: 
    - Captures continious gesture data 1 second per sample.
    - Collects 50 samples for each gesture.
    - Includes a user-friendly guide to start and stop gesture collection.
    - Data is saved as numpy arrays.

2. **`Preprocessor.py`**: 
    - Loads and preprocesses the collected and augmented gesture data.
    - Normalizes the velocities and angles, creates sequences for gesture frames, encodes labels, and splits the data into training, validation, and test sets.

3. **`Evaluation.py`** (Theres multiple evaluation files all with the purpose of testing): 
    - Generates a classification report and confusion matrix.
    - Visualizes the confusion matrix using a heatmap.

4. **`NNModel.py`**: 
    - Defines and trains the Dense Neural Network model for gesture recognition (Classification).
    - Includes model architecture, loss function, optimizer, and training steps.
  
5.  **`GUI.py`**
    - Apon running all prerequisite scripts, (1 through 7) a .h5 file will be created along side a csv file thats used for determining the gesture classes that the model can recognize


## Installation

### Prerequisites

To run the scripts in this project, you need to have the following dependencies installed:

- Python 3.6+ (Im using 3.9 for optimum use with open cv)
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (for model training and evaluation)

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

### 3. Data Shufflling

Run **`DataShuffle.py`**, this step is optional, but will help in training to reduce chances of overfit.

```bash
python DataShuffle.py
```
This will shuffle the samples in your csv file

### 4. Data Preprocessing

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
├── DataShuffle.py                    # Script for data shufflling
├── DataPreprocessor.py               # Script to preprocess the data
├── ModelEvaluation.py                # Script to evaluate the model
├── LSTMGestureClassificationModel.py # Script to define and train the LSTM model
├── GestureRecognition.py               # Script for that recognizes live gestures
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
