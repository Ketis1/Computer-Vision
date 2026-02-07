# Head-Tilt Scrolling

A hands-free computer interaction tool that allows you to scroll documents and web pages by simply tilting your head up or down.

## Features
- **Vertical Scrolling**: Look up to scroll up, look down to scroll down.
- **Hands-Free Control**: Ideal for reading long documentation while drinking coffee or multitasking.
- **Premium UI**: Real-time feedback with a sleek, dark-themed overlay showing tilt intensity and status.
- **Calibration-Free**: Uses normalized face landmarks to adapt to different distances from the camera.

## How it Works
The script uses **MediaPipe Face Mesh** to track the relative position of the nose tip between the forehead and chin. 
- When your nose moves above the center of your face (looking up), it triggers a scroll-up event.
- When it moves below the center (looking down), it triggers a scroll-down event.
- Includes a "Neutral Zone" to prevent accidental scrolling while looking straight ahead.

## Controls
- `p`: Pause/Resume scrolling.
- `q`: Quit the application.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI

## Usage
Run the script using the project's virtual environment:
```powershell
python head-tilt-scrolling/head_tilt_scrolling.py
```
