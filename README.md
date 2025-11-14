# Face Feature Detector

A simple Python application that detects facial features (eyes, nose, mouth) and identifies facial expressions in real-time using your webcam.

## Features

- Detects facial landmarks (eyes, nose, mouth)
- Identifies facial expressions (smiling, neutral, surprised, squinting)
- Real-time processing with webcam feed
- Visual indicators for facial features

## Requirements

- Python 3.6 or higher
- OpenCV
- MediaPipe

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install opencv-python mediapipe
```

## Usage

Run the application:

```bash
python FaceDetector.py
```

The application will open your webcam and start detecting faces. You will see:
- Green dots for eyes
- Blue dots for nose
- Red dots for mouth
- A bounding box around detected faces
- Expression label at the top of the bounding box

Press 'q' to quit the application.

## How It Works

- Uses MediaPipe's Face Detection for face localization
- Uses MediaPipe's Face Mesh for detailed facial landmarks
- Estimates expressions by analyzing distances between facial features
- Visualizes detected features with colored dots