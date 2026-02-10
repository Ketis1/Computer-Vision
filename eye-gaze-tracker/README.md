# Eye Gaze Tracker

This project implements a hands-free cursor control system using eye-gaze tracking. It accurately estimates where you are looking on the screen based on your eye movements and maps those coordinates to your monitor.

## Features
- **Iris Tracking**: Utilizes MediaPipe Face Mesh with refined iris landmarks for high-precision eye tracking.
- **Polynomial Calibration**: Features a 9-point calibration process that builds a second-degree polynomial regression model to handle non-linear eye movement and screen perspective.
- **On-Demand Control**: Control is active only while holding the **'M'** key to prevent accidental cursor movement (the "Midas Touch" problem).
- **Visual Feedback**: Includes a real-time visualization of the gaze point and a debug view of the eye landmarks.
- **Smoothing**: Employs an Exponential Moving Average (EMA) filter to ensure smooth cursor motion.

## Configuration
- **EMA Alpha**: Adjust the `ema_alpha` value in `main.py` (range 0-1) to control the balance between responsiveness and smoothness.
- **Calibration Points**: The system uses a 3x3 grid for the best accuracy.

## Usage
1.  Run the script:
    ```bash
    python main.py
    ```
2.  If no calibration data is found, the **Calibration** process will start automatically.
3.  Look at the **green dots** as they appear on the screen and hold your gaze until they fill up.
4.  Once calibrated, sit in your natural position.
5.  To move the mouse cursor, hold the **'M'** key and look at different parts of the screen.
6.  Press **'C'** at any time to recalibrate, or **'ESC'** to exit.
