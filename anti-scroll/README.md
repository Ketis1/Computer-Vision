# Anti-Scroll System

A focus-assistance tool designed to combat "doom-scrolling" or phone distraction while working at a computer. 

## Purpose
The script monitors your head pose in real-time. If it detects that you are looking down (at a phone or lap) for more than **1.5 seconds**, it triggers a persistent visual alert on your screen to remind you to get back to work.

## How It Works
1. **Head Pose Estimation**: Uses MediaPipe Face Mesh to calculate the pitch (vertical tilt) of your head.
2. **Threshold Monitoring**: If the pitch drops below -15 degrees, a timer starts.
3. **Alert**: If the head remains tilted down for the threshold duration, a window pops up playing a repetitive alert video until you look back at the screen.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Installation
Ensure you have the required libraries installed:
```bash
pip install opencv-python mediapipe numpy
```

## Usage
1. Run the script:
   ```bash
   python anti_scroll.py
   ```
2. The system will start monitoring. If you look down for too long, the alert will trigger.
3. Press **'q'** in the camera preview window to exit.
