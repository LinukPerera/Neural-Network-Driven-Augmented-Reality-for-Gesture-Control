# Neural-Network-Driven-Augmented-Reality-for-Gesture-Control

## There's 2 Proposed ML Methodologies

### Method 1 Gesture Recognition Using CNN (Convolutional Neural Network) and LSTM (Long Short-Term Memory) Networks

This approach aims to capture key points in a human hand to develop a gesture recognition system using HPE (Hand Pose Estimation) techniques employing a CNN (Convolutional Neural Network) and LSTM networks.

### Method 2 Gesture Recognition Using CNN (Convolutional Neural Network) and Dense NN (Neural Network) and Semi-Supervised Learning (Pipeline of Models)

This approach captures weighted human hand velocity data using Hand Pose Estimation (HPE) techniques with MediaPipe Hands, processes it through a Convolutional Neural Network (CNN) for feature extraction, and employs a Dense Neural Network for gesture classification. Additionally, semi-supervised learning methods (Self-Training and Mean Teacher) are used to leverage unlabeled data, improving model robustness and accuracy. This method is lightweight, accurate, and optimized for real-time gesture recognition, with specific handling for swapped "Swipe Left" (rightward motion) and "Swipe Right" (leftward motion) classifications and **recoil filtering**.


## Project Overview

### Method 1 consists of several Python scripts that work together to:

1. **Collect gesture data** using a camera.
2. **Augment** the collected data to increase the dataset's size and diversity.
3. **Shuffle** the now Augmented data.
4. **Preprocess** the data for training, including normalizing and creating sequences.
5. **Train an LSTM model** for gesture recognition.
6. **Evaluate the model** and visualize the results, including a classification report and confusion matrix.

The dataset is collected at 15 frames per sample, with 100 samples per gesture. Data augmentation multiplies the dataset by 25 times by applying different transformations (shifting, scaling, etc.).

### Method 2 consists of several Python scripts that work together to:

1. **Collect gesture data** using a camera.
2. **Evaluate the recorded data** and visualize the results.
3. **Preprocess** the data for training, including normalizing and creating sequences.
4. **Evaluate the Preprocessed data** and visualize the results.
5. **Train NN and Semi-Supervised models** for gesture recognition and recoil filtering.
6. **Evaluate the model** and visualize the results, including a classification report and confusion matrix.

The dataset is collected continuously (e.g., 1 seconds per sample, 50 samples per gesture), saved as NumPy arrays, and preprocessed to include features like absolute velocities, directions, and movement angles. Semi-supervised training uses unlabeled data to enhance performance, with recoil filtering to handle unintended movements.

## Project Components

### The method 1 contains the following scripts:

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


### The Method 2 pipeline includes the following scripts:

1. **`data_collection_NN.py`**:
   - Captures continuous gesture data (1 second per sample, 50 samples per gesture) using MediaPipe Hands.
   - Collects weighted hand velocities for each gesture.
   - Saves data as NumPy arrays (`training_data.npy`, `labels.npy`).

2. **`Data_collection_Semi_Supervised.py`**:
   - Collects gesture data with recoil detection and swapped "Swipe Left" (rightward, `velocity_x > 1000`) and "Swipe Right" (leftward, `velocity_x < -1000`) thresholds.
   - Records 15 samples per gesture (3 seconds each) with a user-friendly interface.
   - Saves unlabeled data as `unlabeled_gesture_data.npy` for semi-supervised training.

3. **`Preprocessor.py`**:
   - Preprocesses collected data for supervised training.
   - Normalizes velocities and angles, creates sequences, and splits data into training, validation, and test sets.
   - Outputs preprocessed data as `processed_training_data_with_angles.npy`.

4. **`Preprocessor_semi.py`**:
   - Preprocesses unlabeled data from `Data_collection_Semi_Supervised.py` for semi-supervised training.
   - Normalizes features and filters outliers.
   - Outputs `preprocessed_unlabeled_data.npy`.

5. **`npy_to_csv_for_semi.py`**:
   - Converts NumPy arrays (e.g., `processed_training_data_with_angles.npy`) to CSV format for compatibility with semi-supervised training and analysis.

6. **`NNModel.py`**:
   - Defines and trains a Dense Neural Network for supervised gesture classification.
   - Uses features like absolute velocities, directions, and angles.
   - Saves the trained model as `gesture_classification_model.h5` (This is needed to train both Semi-Supervised models).

7. **`self_training_model_train.py`**:
   - Implements Self-Training, a semi-supervised method that uses pseudo-labels for high-confidence unlabeled data.
   - Incorporates recoil filtering (velocity < 300, opposite direction).
   - Saves the trained model as `self_training_recoil_model.h5`.

8. **`Mean_Teacher_model_train.py`**:
   - Implements the Mean Teacher semi-supervised method, using a teacher-student model with consistency loss.
   - Includes recoil filtering and swapped left/right thresholds.
   - Saves the trained model as `mean_teacher_recoil_model.h5`.

9. **`Evaluation.py`**:
   - Evaluates the supervised model (`gesture_classification_model.h5`) on test data.
   - Generates classification reports and confusion matrices.

10. **`PreProcessedEvaluation.py`**:
    - Evaluates preprocessed data quality, visualizing feature distributions and recoil detection.

11. **`Model_evaluate_semi-supervised.py`**:
    - Evaluates semi-supervised models (`self_training_recoil_model.h5`, `mean_teacher_recoil_model.h5`) on test data.
    - Generates classification reports, confusion matrices, and comparative performance plots.

12. **`model_test.py`**:
    - Tests models on specific test cases or subsets of data for debugging or validation.

13. **`plotClasses.py`**:
    - Visualizes class distributions or gesture features (e.g., velocity vs. angle scatter plots) to analyze data balance.

14. **`GUI_NN_model.py`**:
    - Provides a real-time GUI for gesture recognition using the supervised model (`gesture_classification_model.h5`).
    - Displays predictions with a darkened overlay and hand landmarks.

15. **`gui_Semi_models.py`**:
    - Real-time GUI for gesture recognition using semi-supervised models (`self_training_recoil_model.h5` or `mean_teacher_recoil_model.h5`).
    - Incorporates recoil filtering and swapped left/right thresholds for accurate classification.

### Installation

#### Prerequisites

To run the scripts, install the following dependencies:

- Python 3.6+ (3.9 recommended for compatibility with OpenCV and TensorFlow I have used Python 3.9.18)
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- scikit-learn (`scikit-learn`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)
- TensorFlow (`tensorflow`, for model training and evaluation)

Install dependencies using pip:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## Usage

### Method 1

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



### Method 2

## 1. Gesture Data Collection

Run either of the following scripts to collect gesture data:

```bash
python data_collection_NN.py
python Data_collection_Semi_Supervised.py
```

- Follow on-screen prompts to perform gestures: `Swipe Up`, `Swipe Down`, `Swipe Right`, `Swipe Left`.
- Data is saved as NumPy arrays:
  - `training_data.npy`
  - `unlabeled_gesture_data.npy`

---

## 2. Data Preprocessing

Preprocess the data for model training:

```bash
python Preprocessor.py
python Preprocessor_semi.py
```

- Outputs:
  - `processed_training_data_with_angles.npy`
  - `preprocessed_unlabeled_data.npy`

Optional: Convert `.npy` files to `.csv`:

```bash
python npy_to_csv_for_semi.py
```

---

## 3. Train Models

Train the **supervised Dense Neural Network** model:

```bash
python NNModel.py
```

Train **semi-supervised** models:

```bash
python self_training_model_train.py
python Mean_Teacher_model_train.py
```

- Output models:
  - `gesture_classification_model.h5`
  - `self_training_recoil_model.h5`
  - `mean_teacher_recoil_model.h5`

---

## 4. Model Evaluation

Evaluate models on the test data:

```bash
python Evaluation.py                      # Supervised model
python Model_evaluate_semi-supervised.py  # Semi-supervised models
python PreProcessedEvaluation.py          # Data quality check
```

- Generates:
  - Classification reports
  - Confusion matrices (`confusion_matrix.png`)
  - Performance comparison plots (`model_comparison_plot.png`)

---

## 5. Real-Time Gesture Recognition

Run the real-time GUI interface:

```bash
python GUI_NN_model.py       # Supervised model
python gui_Semi_models.py    # Semi-supervised models
```

- Displays:
  - Live gesture predictions
  - Hand landmarks
  - Modern UI overlay
- Press `q` to exit.

---

### Example Files (Method 1)

- **Gesture Data (CSV)**: Data captured from **`GestureCollection.py`**.
- **Augmented Data (CSV)**: Data after applying transformations from **`DataAugmentation.py`**.
- **Preprocessed Data (Numpy `.npy` files)**: Data ready for training.
- **Model**: The trained LSTM model (`gesture_lstm_model.h5`).
- **Evaluation Results**: Classification report and confusion matrix heatmap.
- **Collected Data (Numpy `.npy` files)**: Data captured from **`training_data.npy`**.
- **Label Data (Numpy `.npy` files)**: Data captured from **`labels.npy`**.
- **Preprocessed Data (Numpy `.npy` files)**: Data ready for training from **`processed_training_data_with_angles.npy`**.

---

## 📁 Example Files (Method 2)

| Type               | Files                                                                 |
|--------------------|-----------------------------------------------------------------------|
| Collected Data     | `training_data.npy`, `unlabeled_gesture_data.npy`                    |
| Preprocessed Data  | `processed_training_data_with_angles.npy`, `preprocessed_unlabeled_data.npy` |
| Converted Data     | CSV files from `npy_to_csv_for_semi.py` (e.g., `processed_data.csv`) |
| Trained Models     | `gesture_classification_model.h5`, `self_training_recoil_model.h5`, `mean_teacher_recoil_model.h5` |
| Evaluation Results | `confusion_matrix.png`, `model_comparison_plot.png`, classification reports |

---

## File Structure

### ML Method 1
```
.
├── GestureCollection.py                  # Script to collect gesture data
├── DataAugmentation.py                   # Script for data augmentation
├── DataShuffle.py                        # Script for data shufflling
├── DataPreprocessor.py                   # Script to preprocess the data
├── ModelEvaluation.py                    # Script to evaluate the model
├── LSTMGestureClassificationModel.py     # Script to define and train the LSTM model
├── GestureRecognition.py                 # Script for that recognizes live gestures
├── augmented_gesture_data.csv            # Example of collected and augmented data
├── gesture_lstm_model.h5                 # Trained LSTM model (after training)
├── label_encoder.npy                     # Label encoder (for evaluation)
└── confusion_matrix.png                  # Confusion matrix heatmap (for evaluation)
```

### ML Pipelined Method 2

```
.
├── data_collection_NN.py                      # Collects gesture data for supervised training
├── Data_collection_Semi_Supervised.py         # Collects gesture data with recoil and swapped thresholds
├── Preprocessor.py                            # Preprocesses data for supervised training
├── Preprocessor_semi.py                       # Preprocesses unlabeled data for semi-supervised training
├── npy_to_csv_for_semi.py                     # Converts NumPy arrays to CSV
├── NNModel.py                                 # Trains supervised Dense NN model
├── self_training_model_train.py               # Trains Self-Training semi-supervised model
├── Mean_Teacher_model_train.py                # Trains Mean Teacher semi-supervised model
├── Evaluation.py                              # Evaluates supervised model
├── PreProcessedEvaluation.py                  # Evaluates preprocessed data quality
├── Model_evaluate_semi-supervised.py          # Evaluates semi-supervised models
├── model_test.py                              # Tests models on specific cases
├── plotClasses.py                             # Visualizes class distributions/features
├── GUI_NN_model.py                            # GUI for supervised model recognition
├── gui_Semi_models.py                         # GUI for semi-supervised model recognition
├── LICENSE                                    # MIT License file
├── training_data.npy                          # Example collected data (supervised)
├── unlabeled_gesture_data.npy                 # Example collected data (semi-supervised)
├── processed_training_data_with_angles.npy    # Example preprocessed data
├── preprocessed_unlabeled_data.npy            # Example preprocessed unlabeled data
├── gesture_classification_model.h5            # Trained supervised model
├── self_training_recoil_model.h5              # Trained Self-Training model
├── mean_teacher_recoil_model.h5               # Trained Mean Teacher model
├── confusion_matrix.png                       # Example confusion matrix heatmap
└── model_comparison_plot.png                  # Example comparative performance plot
```



## Model Performance

After training and evaluating the model, you'll have access to:
- **Classification report**: A detailed report of precision, recall, f1-score, and support for each gesture class.
- **Confusion matrix heatmap**: A visual representation of the model’s confusion matrix, showing true vs predicted labels.
- The LSTM Model prooved to Overfit however, the NN model had an accuracy of 84.78% when tested against an independent dataset and the self-learning Semi-Supervised model had an accuracy of 69.57% against an independent dataset, and the Student-Teacher Semi-Supervised model had an accuracy of 63.04% against an independent dataset. The mentioned results prooves and confirms the success of the suggested methodology to identify **Instinctive Gestures** leveraging Neural Networks and Machine Learning.

## Contributing

If you have suggestions or improvements for this project, feel free to open an issue or submit a pull request. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
