# MoodMoji: AI Emotion-to-Emoji Personality Mirror

![MoodMoji Banner](emojis/all_emojis.png)

## Overview

MoodMoji is an AI-powered application that analyzes facial expressions in real-time and transforms them into personalized emoji reflections. The application uses computer vision and deep learning to detect faces, recognize emotions, and generate custom emojis that mirror your emotional state.

### Key Features

- **Real-time Emotion Detection**: Recognizes 7 different emotional states (Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral)
- **Custom Emoji Generation**: Creates personalized emojis based on your face and detected emotion
- **Interactive UI**: User-friendly interface with live webcam feed and emoji display
- **Emotion Confidence Meter**: Visual indication of the AI's confidence in its emotion prediction
- **Capture & Save**: Save your custom MoodMojis as image files
- **Demo Mode**: Try the interface without a trained model

## Installation

### Prerequisites

- macOS (tested on macOS with Intel chip)
- Python 3.8 or higher
- Webcam

### Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/MoodMoji.git
   cd MoodMoji
   ```

2. **Create a virtual environment**:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install setuptools wheel && pip install -r requirements.txt
   ```

4. **Download face detection model**:
   ```
   mkdir -p models
   curl -o models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
   curl -o models/res10_300x300_ssd_iter_140000.caffemodel https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
   ```

## Project Structure

- **face_detector.py**: Handles face detection in images and video streams
- **model.py**: Defines the emotion recognition CNN architecture
- **data_preparation.py**: Processes the FER2013 dataset for training
- **train.py**: Trains the emotion recognition model
- **emoji_mapper.py**: Maps detected emotions to emoji representations
- **app.py**: Main application with UI for real-time emotion detection
- **evaluate.py**: Evaluates model performance and generates metrics
- **utils.py**: Helper functions used across the project

## Usage

### Running the Application

To start the MoodMoji application:

```
python app.py
```

Optional arguments:
- `--demo_mode`: Run without a trained model, cycling through emotions
- `--auto_start`: Automatically start webcam on launch

### Using the Interface

1. Click "Start Webcam" to begin emotion detection
2. Position your face in the video feed
3. The application will detect your emotions and display corresponding emojis
4. Use "Capture MoodMoji" to save custom emojis to the `captured_emojis` folder

## Training the Model

### 1. Prepare the Dataset

MoodMoji uses the FER2013 dataset for training. You can download it from Kaggle: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

After downloading, place the `fer2013.csv` file in the `data` directory.

To prepare the dataset:

```
python data_preparation.py
```

### 2. Train the Model

Train the emotion recognition model with:

```
python train.py
```

Optional arguments:
- `--epochs [NUMBER]`: Number of training epochs (default: 50)
- `--batch_size [NUMBER]`: Batch size for training (default: 64)
- `--learning_rate [NUMBER]`: Learning rate (default: 0.001)
- `--lightweight`: Use lightweight model architecture for faster training
- `--augmentation`: Enable data augmentation for better generalization

The trained model will be saved in the `models` directory.

### 3. Evaluate the Model

Evaluate the trained model's performance:

```
python evaluate.py
```

Optional arguments:
- `--model_path [PATH]`: Path to the model file (default: models/emotion_model.h5)
- `--show_examples`: Display examples of correct and incorrect predictions
- `--test_image [PATH]`: Test the model on a specific image

## Customizing MoodMoji

### Creating Custom Emojis

To add your own emoji designs:
1. Create PNG images with transparent backgrounds
2. Name them according to emotions: angry.png, disgust.png, fear.png, happy.png, sad.png, surprise.png, neutral.png
3. Place the files in the `emojis` directory

### Adjusting Model Parameters

For different performance characteristics:
- Use `--lightweight` during training for faster inference
- Modify the CNN architecture in `model.py` for different accuracy/speed tradeoffs
- Adjust emotion stability threshold in `app.py` to change how quickly emotions switch

## Troubleshooting

### Camera Access Issues

If you experience camera permission issues:
1. Go to System Preferences → Security & Privacy → Privacy → Camera
2. Ensure Terminal or Python is allowed access
3. If not listed, run `tccutil reset Camera` in Terminal
4. Restart the application

### Model Performance

If emotion recognition accuracy is low:
1. Train for more epochs with `--epochs 100`
2. Enable data augmentation with `--augmentation` 
3. Try adjusting the learning rate with `--learning_rate 0.0001`

## Credits

- Emotion recognition based on FER2013 dataset
- Face detection using OpenCV's pre-trained models
- Built with TensorFlow, OpenCV, and Tkinter

## License

This project is licensed under the MIT License - see the LICENSE file for details.