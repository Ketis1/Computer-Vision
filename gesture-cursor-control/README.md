# Gesture Cursor Control

A computer vision-based cursor control system that allows you to control your mouse using hand gestures. 

## Gestures
- **Movement**: Open Palm (5 fingers). The cursor follows the center of your palm.
- **Left Click**: Close your hand into a **Fist** and hold for **0.5 seconds**. A progress bar will show the click status.
- **Dragging**: Extend **4 fingers** (excluding the thumb) and hold for **0.3 seconds**. The system will "grab" the object until you release the gesture.
- **Release (Stability)**: There is a 0.2-second grace period when releasing a drag to prevent accidental "drops" if the hand momentarily disappears.

## Features
- **High Sensitivity**: Optimized mapping margins (10% on sides/bottom, 20% on top) to ensure easy reach of all screen corners.
- **Real-time Feedback**: Status messages and progress bars are drawn directly on the camera feed.
- **Resizable Preview**: The camera window is resizable to fit your workstation layout.
- **Smoothing**: Implemented moving average filter to reduce cursor jitter.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- NumPy (`numpy`)

## Installation
Install dependencies via pip:
```bash
pip install opencv-python mediapipe pyautogui numpy
```

## Usage
Run the script:
```bash
python gesture_cursor_control.py
```
Press **'q'** in the camera window to quit the application.

### Fail-Safe
This script uses PyAutoGUI's default fail-safe mechanism. If you move the mouse manually to any corner of the screen, the script will crash/exit to prevent loss of control.
