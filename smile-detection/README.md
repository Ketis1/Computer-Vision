# Smile Detection

A simple and fun experiment that reacts to your facial expressions.

## Purpose
Detects when a user is smiling and swaps a "reaction image" accordingly. It serves as a demonstration of facial landmark normalization and ratio calculation for expression detection.

## How It Works
- **Landmark Tracking**: Uses MediaPipe Face Mesh to track 468 3D face landmarks.
- **Smile Ratio**: Calculates the ratio between the mouth width and the distance between eyes. This normalization ensures the detection works regardless of how close or far you are from the camera.
- **Reaction**: Displays a "Happy" cat image when a smile is detected and a "Serious" cat image otherwise.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)

## Installation
Install dependencies:
```bash
pip install opencv-python mediapipe
```

## Usage
1. Run the script:
   ```bash
   python smile_detection.py
   ```
2. Two windows will open: the camera feed and the "Reaction" window.
3. Smile to see the reaction change!
4. Press **'q'** to quit.
