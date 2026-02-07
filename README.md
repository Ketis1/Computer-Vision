# Computer-Vision Projects

Welcome to my collection of Computer Vision scripts and mini-projects. This repository serves as a playground for various CV implementations using libraries like OpenCV, MediaPipe, and Ultralytics (YOLO).

## Overview
While the primary focus of these tools is productivity and effective learning (helping you stay focused, maintain good posture, or study more efficiently), you will also find entertainment experiments and fun CV applications here.

---

## Projects Gallery

### Productivity & Learning
*   **[Anki Hand Gesture Control](./anki-gesture-control)**
    *   **Description**: Control Anki Desktop touchlessly using hand gestures. Map 1-5 fingers to Review actions (Again, Hard, Good, Easy, Show Answer).
    *   **Key Tech**: MediaPipe Hands, AnkiConnect API.
    *   **Documentation**: [See README](./anki-gesture-control/README.md)
*   **[Anti-Scroll System](./anti-scroll)**
    *   **Description**: Combats the "doom-scrolling" habit. It detects when your head is tilted down (looking at a phone) and triggers a visual alert window to bring your focus back to the screen.
    *   **Key Tech**: MediaPipe Face Mesh (Head Pose Estimation).
    *   **Documentation**: [See README](./anti-scroll/README.md)
*   **[Phone Detection](./phone-detection)**
    *   **Description**: A real-time object detection script that looks for smartphones in the camera frame. Perfect for keeping your desk a "phone-free zone" while working.
    *   **Key Tech**: YOLOv8 (Ultralytics).
    *   **Documentation**: [See README](./phone-detection/README.md)
*   **[Gesture Cursor Control](./gesture-cursor-control)**
    *   **Description**: Control your mouse cursor using hand gestures. Move your palm to move the cursor, fist to click, and 4 fingers to drag.
    *   **Key Tech**: MediaPipe Hands, PyAutoGUI, OpenCV.
    *   **Documentation**: [See README](./gesture-cursor-control/README.md)
*   **[Posture Detection](./posture-detection)**
    *   **Description**: Monitors your sitting posture. After a quick calibration, it alerts you if you start slouching by measuring the vertical distance between your ears and shoulders.
    *   **Key Tech**: MediaPipe Pose.
    *   **Documentation**: [See README](./posture-detection/README.md)
*   **[Head-Tilt Scrolling](./head-tilt-scrolling)**
    *   **Description**: Hands-free document scrolling. Look up to scroll up, look down to scroll down. Perfect for reading documentation while multitasking.
    *   **Key Tech**: MediaPipe Face Mesh, PyAutoGUI.
    *   **Documentation**: [See README](./head-tilt-scrolling/README.md)

### Entertainment & Experiments
*   **[Smile Detection](./smile-detection)**
    *   **Description**: A fun project that monitors your facial expression. It displays different reaction images based on whether you are smiling or staying serious.
    *   **Key Tech**: MediaPipe Face Mesh (Mouth/Eye Ratio).
    *   **Documentation**: [See README](./smile-detection/README.md)

---

## Getting Started
Most projects in this repo require Python 3.x and a working webcam.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ketis1/Computer-Vision.git
    cd Computer-Vision
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some projects (like Phone Detection) might require additional large files like YOLO weights (`.pt`).*

3.  **Explore**: Navigate to a specific project folder and run the `.py` script!

---

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
