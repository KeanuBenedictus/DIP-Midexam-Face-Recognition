# Face Feature Detector

A simple Python application that detects facial features (eyes, nose, mouth) and identifies facial expressions in real-time using your webcam.

## Features

- Detects facial landmarks (eyes, nose, mouth)
- Identifies facial expressions (smiling, neutral, surprised, squinting)
- Real-time processing with webcam feed
- Visual indicators for facial features
- Drowsiness detection with audio and visual alerts
- Recording functionality with start/stop toggle
- Real-time clock display in the format DD/MM/YYYY-HH.MM.SS
- Custom recording filenames with start and end timestamps

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
- Real-time clock at the bottom left corner in DD/MM/YYYY-HH.MM.SS format

### Controls:
- Press 'q' to quit the application
- Press 's' to save a snapshot
- Press 'r' to start/stop recording

### Recording:
- When pressing 'r', the recording will start and a file will be created in the 'recorded' directory
- Press 'r' again to stop recording
- The recorded video will be saved with a filename in the format: `recording_[start_datetime]-[end_datetime].avi`
- The real-time clock is visible in both the live feed and the recorded video

### Drowsiness Detection:
- The application monitors for signs of drowsiness based on eye aspect ratio
- If drowsiness is detected, audio alerts (beep) and visual alerts (flashing screen) will be triggered

## How It Works

- Uses MediaPipe's Face Detection for face localization
- Uses MediaPipe's Face Mesh for detailed facial landmarks
- Estimates expressions by analyzing distances between facial features
- Visualizes detected features with colored dots
- Implements recording functionality using OpenCV's VideoWriter
- Displays real-time clock using datetime module