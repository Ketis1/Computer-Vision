# Phone Detection

A real-time object detection tool specifically tuned to identify smartphones within the camera frame.

## Purpose
Helps maintain a "phone-free zone" by detecting when a mobile device is present on your desk or in your hands. It provides both visual feedback on the camera feed and a console alert.

## How It Works
- **Object Detection**: Leverages the YOLOv8 (You Only Look Once) architecture for high-speed detection.
- **Optimization**: Uses the 'nano' version of YOLOv8 (yolov8n) and reduced image size for high performance even on standard CPUs.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)

## Installation
1. Install dependencies:
   ```bash
   pip install opencv-python ultralytics
   ```
2. The script will use the local `yolov8n.pt` weights file if present, or download them automatically if missing.

## Usage
1. Run the script:
   ```bash
   python phone_detection.py
   ```
2. The camera feed will show a red box around any detected phones with a confidence score.
3. Press **'q'** to quit.
