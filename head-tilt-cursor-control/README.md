# Head Tilt Cursor Control

Implement a virtual "head joystick" to control your mouse cursor using head tilts. This system functions like a trackpoint, where the degree of head tilt determines the speed of cursor movement.

## Features
- **3D Head Pose Estimation**: Uses MediaPipe Face Mesh and OpenCV's `solvePnP` to estimate Pitch and Yaw angles in real-time.
- **Trackpoint Logic**: Movement is based on relative tilt from a neutral position. The further you tilt, the faster the cursor moves.
- **Deadzone Shield**: A configurable deadzone prevents cursor drift from minor head movements.
- **Blink-to-Click**: Hold your **right eye** (mirrored on screen) closed for **1 second** to trigger a left mouse click. A visual progress bar indicates the click timing.
- **Angular Normalization**: Handles the 180-degree boundary correctly to prevent "jumping" when tilting far in any direction.
- **On-Demand Control**: Mouse movement is active only while holding the **Right CTRL** key.

## Configuration
- **Deadzone**: Adjust `self.deadzone_x` and `self.deadzone_y` in `main.py` to change sensitivity to small movements.
- **Speed Multipliers**: `self.speed_multiplier_x` and `self.speed_multiplier_y` allow independent control over horizontal and vertical cursor speeds.

## Usage
1.  Run the script:
    ```bash
    python main.py
    ```
2.  Sit in a comfortable, neutral position.
3.  Press **'C'** to calibrate your neutral center.
4.  Hold the **Right CTRL** key to start moving the cursor with head tilts.
5.  To click, close your right eye and hold it for 1 second until the progress bar completes.
6.  Press **'ESC'** to exit.
