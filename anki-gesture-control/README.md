# Anki Hand Gesture Control

This project enables touchless control of **Anki Desktop** using a webcam and computer vision technology. By leveraging the MediaPipe library, the script detects the number of extended fingers and sends corresponding commands to Anki via the AnkiConnect API.

## Purpose
The main goal is to increase comfort during study sessions. It allows for:
- Studying in a more comfortable position (no need to keep hands on the keyboard/mouse).
- Reviewing flashcards when your hands are busy.
- A modern and interactive way to engage with the software.

## How It Works
The script launches a webcam preview and analyzes hand landmarks in real-time.
1. **Detection:** The system counts how many fingers are extended.
2. **Confirmation:** To avoid accidental triggers, a gesture must be held for **1.5 seconds** (a green progress bar will appear).
3. **Action:** Once confirmed, the script sends an HTTP request to the AnkiConnect add-on, which performs the action in Anki.
4. **Cooldown:** After an action is triggered, there is a short cooldown (1.5s) to prevent multiple triggers of the same command.

## Gesture Mapping
| Gesture (Finger count) | Anki Action | Description |
| :--- | :--- | :--- |
| **1 Finger (Index)** | Again | Rating: 1 |
| **2 Fingers** | Hard | Rating: 2 |
| **3 Fingers** | Good | Rating: 3 |
| **4 Fingers** | Easy | Rating: 4 |
| **5 Fingers (Open Hand)** | Show Answer | Reveals the answer |

## Configuration

### 1. System Requirements
- Python 3.x
- A working webcam

### 2. Library Installation
Run the following command to install the required packages:
```bash
pip install opencv-python mediapipe requests
```

### 3. Anki Configuration (AnkiConnect)
The script communicates with Anki via the **AnkiConnect** add-on.
1. Open Anki Desktop.
2. Go to `Tools` -> `Add-ons` -> `Get Add-ons...`.
3. Enter the code: **2055492159**.
4. Restart Anki after installation.

**Important (CORS/Security):** 
Ensure AnkiConnect allows incoming connections. 
- Go to `Tools` -> `Add-ons`, select **AnkiConnect**, and click **Config**.
- Ensure `"http://localhost"` is in the `webBindOriginList` section. If you encounter connection issues, you can add `"*"` (allows all origins, but use with caution).

## How to Run
1. Make sure **Anki Desktop is running**.
2. Run the script:
   ```bash
   python anki_hand_gesture_control.py
   ```
3. Show gestures to the camera, holding them still until the progress bar fills up.
4. To exit, press the **'q'** key in the camera preview window.

## Troubleshooting
- **Connection Error:** Check if Anki is open and if the AnkiConnect add-on is running on port 8765.
- **Poor Detection:** Ensure your hand is well-lit and fully visible in the frame. The script works best when the palm is facing the camera directly.
