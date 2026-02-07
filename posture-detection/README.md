# Posture Detection

A health-oriented tool that monitors your sitting posture and alerts you when you begin to slouch.

## Purpose
Long hours at a desk often lead to poor posture. This tool helps you maintain an upright position by providing immediate visual feedback when your posture deviates from a calibrated "good" state.

## How It Works
1. **Calibration**: Upon startup, you sit in your ideal straight posture and press 'c'. The system records the vertical distance between your ears and shoulders.
2. **Monitoring**: Uses MediaPipe Pose to track your landmarks in real-time.
3. **Alert**: If the ear-to-shoulder distance drops below 85% of your calibrated ideal, the system displays a "STRAIGHTEN UP!" warning.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Installation
Install the necessary packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage
1. Run the script:
   ```bash
   python posture_detection.py
   ```
2. Sit straight and press **'c'** to calibrate the system.
3. If you slouch, the text on the screen will turn red and warn you.
4. Press **'q'** to quit.
