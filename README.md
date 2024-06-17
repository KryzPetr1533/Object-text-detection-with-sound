# Real-Time Object Detection and OCR with Text-to-Speech

This project captures video from a webcam, detects objects using YOLOv5, performs Optical Character Recognition (OCR) on the video frames using EasyOCR, and uses text-to-speech (TTS) to read out the detected text.

## Features

- Real-time video capture from webcam using OpenCV
- Object detection using YOLOv5
- OCR to detect text in video frames using EasyOCR
- Text-to-speech using pyttsx3
- Displays detected text on the video frame
- Exits when ESC key is pressed

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- EasyOCR
- Torch
- torchvision
- pyttsx3
- ultralytics (for YOLOv5)
- Pillow (optional for text rendering workaround)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/KryzPetr1533/Object-text-detection-with-sound.git
   cd Object-text-detection-with-sound
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the script**:

   ```bash
   python model.py
   ```

2. **Exit the application**:
   - Press the `ESC` key to exit the application.

## Code Explanation

### Import Libraries

The script imports necessary libraries including OpenCV for video capture, EasyOCR for text recognition, Torch and YOLOv5 for object detection, pyttsx3 for text-to-speech, threading and queue for concurrent text processing, and logging for debug messages.

### Initialize Components

- Initializes the EasyOCR reader.
- Initializes the TTS engine.
- Sets up a queue for managing text to be spoken.
- Opens the video capture device.

### Main Processing Loop

- Captures video frames from the webcam.
- Converts frames to grayscale for OCR processing.
- Detects objects and text within the frames.
- Draws bounding boxes around detected objects and text.
- Adds detected text to the TTS queue if it hasn't been spoken yet.
- Displays the processed video frames.

### Graceful Exit

- Releases the video capture device and closes all OpenCV windows.
- Signals the TTS thread to exit and waits for it to finish.

## Contributing

If you want to contribute to this project, feel free to fork the repository and submit pull requests. Please make sure to follow the coding standards and write clear commit messages.

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR.
- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection.
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech.