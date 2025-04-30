# MoodMoji: AI Emotion-to-Emoji Personality Mirror

MoodMoji is an innovative desktop application that leverages artificial intelligence to detect your emotions in real-time using your webcam. By analyzing your facial expressions, it maps your emotions to corresponding emojis, providing a fun and interactive way to visualize your mood. Additionally, MoodMoji enhances your experience by suggesting Spotify playlists tailored to your detected emotion, allowing you to immerse yourself in music that matches or uplifts your mood.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview
MoodMoji uses a deep learning model to classify emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral from live webcam footage. The app displays the detected emotion as an emoji and suggests a Spotify playlist that aligns with your mood. You can also capture custom emojis that blend your face with the detected emotion emoji, saving them as PNG files for keepsakes or sharing.

The project is built with Python, utilizing libraries like OpenCV for webcam access, TensorFlow for emotion recognition, and Tkinter for the graphical user interface. It’s designed to be user-friendly and engaging, making it a great tool for personal entertainment or exploring AI emotion detection.

## Features
- **Real-time Emotion Detection**: Detects emotions from your facial expressions using a pre-trained deep learning model.
- **Emoji Mapping**: Displays an emoji corresponding to your detected emotion (e.g., 😢 for Sad, 😊 for Happy).
- **Mood Playlist Suggestion**: Provides a clickable link to a Spotify playlist that matches your current emotion (e.g., "Sad Songs" for a sad mood).
- **Custom Emoji Capture**: Allows you to save a custom emoji that overlays your face with the detected emotion emoji.
- **Demo Mode**: If the pre-trained model is unavailable, the app can run in demo mode, cycling through emotions randomly.
- **User-Friendly Interface**: Built with Tkinter, featuring a clean layout with a video feed, emoji display, and controls.

## Screenshots
Here are some snapshots of MoodMoji in action:

- **Detecting Emotion with Playlist Suggestion**  
  ![MoodMoji detecting a sad emotion with playlist suggestion](screenshots/screenshot_1.png)

- **Capturing a Custom Emoji**  
  ![MoodMoji capturing a custom emoji](screenshots/screenshot_2.png)

## Requirements
To run MoodMoji, you’ll need the following:

- **Operating System**: macOS (tested on macOS 14+), Linux, or Windows (may require adjustments for Windows).
- **Hardware**: A webcam for real-time emotion detection.
- **Python**: Version 3.9 or higher (tested with 3.12.7).
- **Dependencies**: The following Python packages (listed in `requirements.txt`):
  - `opencv-python==4.5.5.64`: For webcam access and image processing.
  - `tensorflow==2.16.2`: For the emotion recognition model.
  - `numpy==1.26.4`: For numerical operations.
  - `matplotlib>=3.5.2`: For plotting (used in training scripts).
  - `pillow>=9.1.1`: For image handling in Tkinter.
  - `scikit-learn>=1.1.1`: For data preprocessing (used in training scripts).
  - `tqdm>=4.64.0`: For progress bars during training.
  - `pandas>=1.4.2`: For data handling (used in training scripts).
  - `h5py>=3.7.0`: For loading the pre-trained model.
  - `seaborn>=0.11.2`: For visualization (used in training scripts).
  - `pyobjc-core>=7.3` and `pyobjc-framework-Cocoa>=7.3`: For macOS compatibility with Tkinter.

## Installation
Follow these steps to set up and run MoodMoji on your machine:

1. **Clone the Repository**  
   Clone the project from GitHub:
   ```bash
   git clone https://github.com/tirth8205/MoodMoji.git
   cd MoodMoji
   ```

2. **Create a Virtual Environment**  
   Create a virtual environment to isolate dependencies:
   ```bash
   python3 -m venv .venv_system
   source .venv_system/bin/activate
   ```

3. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **(macOS Only) Ensure Tkinter Compatibility**  
   If you encounter Tkinter issues on macOS, ensure your Python installation includes Tcl/Tk support. You may need to install Python via pyenv with Tcl/Tk:
   ```bash
   brew install tcl-tk
   pyenv uninstall 3.12.7
   env PYTHON_CONFIGURE_OPTS="--with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6' --with-tcltk-includes='-I$(brew --prefix tcl-tk)/include'" pyenv install 3.12.7
   ```
   Then recreate the virtual environment as in step 2.

## Usage
1. **Launch the Application**  
   Run the app with the following command (ensure the virtual environment is activated):
   ```bash
   export TK_SILENCE_DEPRECATION=1
   python3 app.py --auto_start
   ```
   The `--auto_start` flag automatically starts the webcam. Omit it if you prefer to start the webcam manually.

2. **Interact with the App**  
   - The app will open a window displaying your webcam feed on the left.
   - On the right, the "Your MoodMoji" section shows the emoji corresponding to your detected emotion.
   - Below the video feed, the "Controls" section includes:
     - **Start/Stop Webcam**: Toggle the webcam on or off.
     - **Capture MoodMoji**: Save a custom emoji combining your face with the detected emotion.
     - **Demo Mode**: Toggle to cycle through emotions randomly if the model is unavailable.
     - **Emotion and Confidence**: Displays the detected emotion and confidence level.
     - **Mood Playlist**: Shows a Spotify playlist link matching your mood—click to open it in your browser.

3. **Save Custom Emojis**  
   Click the "Capture MoodMoji" button to save a custom emoji to the `captured_emojis` directory. The filename will include the emotion and timestamp (e.g., `moodmoji_sad_20250430_123456.png`).

## Troubleshooting
- **Webcam Not Working**  
  Ensure your webcam is connected and accessible. On macOS, check System Preferences > Security & Privacy > Camera to grant permission to Python.
  
- **Tkinter Errors**  
  If you see `ModuleNotFoundError: No module named '_tkinter'`, your Python installation lacks Tcl/Tk support. Follow the macOS-specific installation steps above to reinstall Python with Tcl/Tk.

- **Emotion Model Not Found**  
  If `models/emotion_model.h5` is missing, the app will run in demo mode. You can train a new model using `train.py` or download a pre-trained model if available.

- **Spotify Playlist Links Not Opening**  
  Ensure you have a web browser installed and that the Spotify links are valid. You can update the links in `app.py` under the `mood_playlists` dictionary if needed.

## Contributing
We welcome contributions to improve MoodMoji! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request with a detailed description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments/docstrings.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions, suggestions, or issues, feel free to reach out:
- **Email**: [tirthkanani18@gmail.com](mailto:tirthkanani18@gmail.com)
- **GitHub Issues**: [https://github.com/tirth8205/MoodMoji/issues](https://github.com/tirth8205/MoodMoji/issues)

## Acknowledgments
- The emotion recognition model is based on a pre-trained deep learning model, fine-tuned for this project.
- Thanks to the open-source community for providing libraries like OpenCV, TensorFlow, and Tkinter.
- Spotify playlist links are sourced from public playlists available on Spotify as of April 2025.
- Special thanks to the contributors and users who provide feedback and suggestions for improving MoodMoji.
```
- This project is inspired by the growing interest in AI and emotion recognition technologies, aiming to create a fun and engaging user experience.
```
- The emoji mapping is based on common emotional associations with emojis, enhancing the relatability of the app.
```
- The custom emoji feature is a playful addition, allowing users to create and share their unique MoodMoji captures.
```
- The demo mode is included to ensure the app remains functional even if the model is unavailable, providing a fallback experience.
```
- The project is a personal endeavor by Tirth Kanani, showcasing the potential of AI in everyday applications and the intersection of technology and emotions.
```
- The app is designed to be a fun tool for exploring emotions and music, encouraging users to reflect on their moods and enjoy the process.
```
- The project is a work in progress, and future updates may include additional features, improved models, and expanded functionality based on user feedback.
```
- The app is intended for personal use and entertainment, and while it provides insights into emotions, it should not be used as a substitute for professional mental health support.
```
- The project is open to collaboration and contributions, inviting developers and enthusiasts to join in enhancing the MoodMoji experience.
```
- The app is a demonstration of the capabilities of AI in real-time applications, showcasing how technology can enhance our understanding of emotions and self-expression.
```
- The project is a reflection of the developer's passion for AI, music, and user experience design, aiming to create a unique and enjoyable application.