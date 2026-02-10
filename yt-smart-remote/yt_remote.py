import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import keyboard
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Settings
GESTURE_COOLDOWN = 1.0  # Seconds between commands
SEQUENCE_TIMEOUT = 1.5  # Seconds to complete Palm > Fist > Palm
SWIPE_THRESHOLD = 0.15  # Minimum horizontal movement for swipe
SWIPE_WINDOW = 10       # Number of frames to track movement

# Premium Colors
COLOR_ACCENT = (255, 100, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (20, 20, 20)

# State Variables
last_command_time = 0
gesture_sequence = []
sequence_start_time = 0
last_feedback_msg = "SITTING IDLE"
feedback_expiry = 0
pos_history = deque(maxlen=SWIPE_WINDOW) # Track hand x-pos
state_history = deque(maxlen=5) # Track last 5 detected states for stability

# Global Toggle State
show_window = True
keep_running = True

def get_hand_state(hand_landmarks):
    # Check if fingers are extended
    fingers = []
    
    # Thumb: Check x-pos relative to IP joint
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    fingers.append(abs(thumb_tip.x - hand_landmarks.landmark[0].x) > abs(thumb_ip.x - hand_landmarks.landmark[0].x))
    
    # Other 4 fingers: 8(index), 12(middle), 16(ring), 20(pinky)
    # Using a more robust check: distance from point 0 (wrist) 
    # and comparing tip to MCP (base) instead of PIP (middle)
    for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y)
    
    count = fingers.count(True)
    
    # Victory (Index + Middle extended, others folded)
    # More robust check: allow slight variations in other fingers if they are still "down"
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "VICTORY"
    # Palm (all or 4 fingers)
    if count >= 4:
        return "PALM"
    # Fist (0 or only thumb)
    if count <= 1 and not fingers[1]:
        return "FIST"
    # Volume (Index only - currently disabled by user request)
    if count == 1 and fingers[1]:
        tip = hand_landmarks.landmark[8]
        pip = hand_landmarks.landmark[6]
        if tip.y < pip.y - 0.05: return "INDEX_UP"
        if tip.y > pip.y + 0.05: return "INDEX_DOWN"
        
    return "UNKNOWN"

def detect_swipe():
    if len(pos_history) < SWIPE_WINDOW:
        return None
    
    start_x = pos_history[0]
    end_x = pos_history[-1]
    diff = end_x - start_x
    
    if diff > SWIPE_THRESHOLD:
        return "RIGHT"
    elif diff < -SWIPE_THRESHOLD:
        return "LEFT"
    return None

def draw_ui(frame, msg, current_state):
    h, w, _ = frame.shape
    # Dark Header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "YT SMART REMOTE", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TEXT, 1)
    
    # Feedback Message
    color = COLOR_ACCENT if time.time() < feedback_expiry else (150, 150, 150)
    cv2.putText(frame, msg, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Legend
    cv2.putText(frame, "Palm > Fist > Palm: Play/Pause", (w-300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "Victory + Swipe L/R: Skip 10s", (w-300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Debug State (Bottom Left)
    cv2.putText(frame, f"STATE: {current_state}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

def toggle_gui():
    global show_window
    show_window = not show_window
    if not show_window:
        cv2.destroyAllWindows()
        print(f"[{time.strftime('%H:%M:%S')}] Switched to HEADLESS MODE (Alt+H)")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Switched to GUI MODE (Alt+H)")

def exit_program():
    global keep_running
    print(f"[{time.strftime('%H:%M:%S')}] Exit triggered via Global Hotkey: Alt+Q")
    keep_running = False

# Register Global Hotkeys
keyboard.add_hotkey('alt+h', toggle_gui)
keyboard.add_hotkey('alt+q', exit_program)

def main():
    global last_command_time, gesture_sequence, sequence_start_time, last_feedback_msg, feedback_expiry, show_window, keep_running
    
    cap = cv2.VideoCapture(0)
    print("YT Smart Remote Active...")
    print("GLOBAL HOTKEYS (Work from anywhere):")
    print(" - Alt + H : Toggle GUI/Headless")
    print(" - Alt + Q : Exit Program and release camera")

    try:
        while cap.isOpened() and keep_running:
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_time = time.time()
            current_state = "NONE"

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    if show_window:
                        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    raw_state = get_hand_state(hand_lms)
                    state_history.append(raw_state)
                    
                    if state_history:
                        valid_states = [s for s in state_history if s != "UNKNOWN"]
                        if valid_states:
                            state = max(set(valid_states), key=valid_states.count)
                        else:
                            state = "UNKNOWN"
                    else:
                        state = "UNKNOWN"
                    
                    current_state = state
                    wrist_x = hand_lms.landmark[0].x
                    pos_history.append(wrist_x)

                    if current_time - last_command_time > GESTURE_COOLDOWN:
                        if state == "PALM":
                            if not gesture_sequence:
                                gesture_sequence = ["PALM"]
                                sequence_start_time = current_time
                            elif len(gesture_sequence) == 2 and gesture_sequence[-1] == "FIST":
                                pyautogui.press('space')
                                last_feedback_msg = "COMMAND: PLAY/PAUSE"
                                print(f"[{time.strftime('%H:%M:%S')}] Triggered: Play/Pause")
                                feedback_expiry = current_time + 1.5
                                last_command_time = current_time
                                gesture_sequence = []
                        elif state == "FIST":
                            if len(gesture_sequence) == 1 and gesture_sequence[0] == "PALM":
                                gesture_sequence.append("FIST")
                        
                        if gesture_sequence and (current_time - sequence_start_time > SEQUENCE_TIMEOUT):
                            gesture_sequence = []

                        if not gesture_sequence and state == "VICTORY" and len(pos_history) >= 5:
                            swipe = detect_swipe()
                            if swipe == "RIGHT":
                                pyautogui.press('l')
                                last_feedback_msg = "COMMAND: SKIP +10s"
                                print(f"[{time.strftime('%H:%M:%S')}] Triggered: Skip Forward")
                                feedback_expiry = current_time + 1.0
                                last_command_time = current_time
                                pos_history.clear()
                            elif swipe == "LEFT":
                                pyautogui.press('j')
                                last_feedback_msg = "COMMAND: SKIP -10s"
                                print(f"[{time.strftime('%H:%M:%S')}] Triggered: Skip Backward")
                                feedback_expiry = current_time + 1.0
                                last_command_time = current_time
                                pos_history.clear()
            else:
                pos_history.clear()

            if show_window:
                msg = last_feedback_msg
                if gesture_sequence:
                    msg = f"SEQUENCE: {' -> '.join(gesture_sequence)}"
                draw_ui(frame, msg, current_state)
                cv2.imshow('YT Smart Remote', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('h'): toggle_gui()
            else:
                # Essential for processing window events even if hidden
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\nProgram stopped by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Cleanup complete.")

if __name__ == "__main__":
    main()
