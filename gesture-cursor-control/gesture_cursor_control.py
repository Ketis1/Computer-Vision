import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# PyAutoGUI configuration
# Disable fail-safe if you want the cursor to be able to reach the very corners
# or keep it True (default) to kill the script by moving mouse to corner.
pyautogui.FAILSAFE = True

def count_fingers(hand_landmarks):
    """
    Counts which fingers are extended.
    Returns: [thumb, index, middle, ring, pinky]
    """
    fingers = []
    
    # Thumb (lateral check)
    # For horizontal movement, we check if the tip is further than the joint.
    # We use x coordinate (MediaPipe landmarks: 4 tip, 3 IP joint)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
        
    # Other 4 fingers (vertical check)
    # landmarks: index(8), middle(12), ring(16), pinky(20)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Gesture Cursor Control', cv2.WINDOW_NORMAL)
    
    # Screen resolution
    screen_w, screen_h = pyautogui.size()
    
    # Smoothing variables
    smooth_factor = 3  # Reduced for more responsiveness (was 5)
    coords_history = []
    
    print("Gesture Cursor Control Started.")
    print("Mapping:")
    print(" - Open Palm (5 fingers) -> Move Cursor")
    print(" - Fist (0 fingers)      -> Click (0.5s hold)")
    print(" - 4 Fingers             -> Drag (0.3s hold)")
    print("Press 'q' in the camera window to quit.")
    
    # State tracking
    is_clicked = False
    is_mouse_down = False
    click_ready_start_time = 0
    grab_ready_start_time = 0
    last_pinch_time = 0 # To stabilize drag release
    CLICK_DELAY = 0.5  # seconds
    GRAB_DELAY = 0.3   # seconds
    DRAG_RELEASE_DELAY = 0.2 # seconds
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1) # Flip for mirror effect
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture_active = False
        target_x, target_y = 0, 0
        status_msg = "CURSOR: IDLE (Victory to move, Pinch to drag)"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                fingers = count_fingers(hand_landmarks)
                # 1. Gesture Definitions
                f_count = sum(fingers)
                is_open_palm = (f_count == 5)
                is_fist = (f_count == 0)
                is_four_fingers = (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0)
                
                # Tracking point: Middle of the hand (landmark 9 - Middle finger MCP)
                palm_center = hand_landmarks.landmark[9]
                avg_x, avg_y = palm_center.x, palm_center.y
                
                if is_four_fingers:
                    gesture_active = True
                    if not is_mouse_down:
                        if grab_ready_start_time == 0: grab_ready_start_time = time.time()
                        hold_duration = time.time() - grab_ready_start_time
                        if hold_duration >= GRAB_DELAY:
                            pyautogui.mouseDown()
                            is_mouse_down = True
                            status_msg = "DRAGGING..."
                        else:
                            status_msg = f"GRABBING... {int((GRAB_DELAY - hold_duration) * 1000)}ms"
                            cv2.rectangle(frame, (10, 70), (int(10 + 200 * (hold_duration/GRAB_DELAY)), 90), (0, 255, 255), -1)
                    else:
                        status_msg = "DRAGGING"
                    
                    last_pinch_time = time.time() # Stabilizer
                    
                    # Map to screen (using asymmetric margins for better reachability)
                    m_x = 0.1
                    m_y_top = 0.2
                    m_y_bottom = 0.1
                    target_x = np.interp(avg_x, [m_x, 1.0 - m_x], [0, screen_w])
                    target_y = np.interp(avg_y, [m_y_top, 1.0 - m_y_bottom], [0, screen_h])
                    pyautogui.moveTo(target_x, target_y, _pause=False)

                    # Grace period for Drag
                    status_msg = "DRAGGING (Holding...)"
                    gesture_active = True
                    m_x = 0.1
                    m_y_top = 0.2
                    m_y_bottom = 0.1
                    target_x = np.interp(avg_x, [m_x, 1.0 - m_x], [0, screen_w])
                    target_y = np.interp(avg_y, [m_y_top, 1.0 - m_y_bottom], [0, screen_h])
                    pyautogui.moveTo(target_x, target_y, _pause=False)

                elif is_open_palm:
                    # Move Cursor
                    if is_mouse_down:
                        pyautogui.mouseUp()
                        is_mouse_down = False
                    
                    gesture_active = True
                    # Map to screen (using asymmetric margins for better reachability)
                    m_x = 0.1
                    m_y_top = 0.2
                    m_y_bottom = 0.1
                    target_x = np.interp(avg_x, [m_x, 1.0 - m_x], [0, screen_w])
                    target_y = np.interp(avg_y, [m_y_top, 1.0 - m_y_bottom], [0, screen_h])
                    
                    # Update smoothing history
                    coords_history.append((target_x, target_y))
                    if len(coords_history) > smooth_factor: coords_history.pop(0)
                    smoothed_x = sum(c[0] for c in coords_history) / len(coords_history)
                    smoothed_y = sum(c[1] for c in coords_history) / len(coords_history)
                    
                    pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)
                    status_msg = "CURSOR: ACTIVE"
                    grab_ready_start_time = 0

                elif is_fist:
                    # Click
                    if is_mouse_down:
                        pyautogui.mouseUp()
                        is_mouse_down = False
                    
                    gesture_active = True
                    if click_ready_start_time == 0: click_ready_start_time = time.time()
                    hold_duration = time.time() - click_ready_start_time
                    
                    if hold_duration >= CLICK_DELAY:
                        if not is_clicked:
                            pyautogui.click()
                            is_clicked = True
                        status_msg = "CLICKED!"
                    else:
                        status_msg = f"CLICKING... {int((CLICK_DELAY - hold_duration) * 1000)}ms"
                        cv2.rectangle(frame, (10, 70), (int(10 + 200 * (hold_duration/CLICK_DELAY)), 90), (255, 0, 255), -1)
                
                else:
                    # Reset
                    if is_mouse_down:
                        pyautogui.mouseUp()
                        is_mouse_down = False
                    is_clicked = False
                    click_ready_start_time = 0
                    grab_ready_start_time = 0
                    coords_history = []
        else:
            if is_mouse_down:
                pyautogui.mouseUp()
                is_mouse_down = False
            coords_history = []
            is_clicked = False
            click_ready_start_time = 0
        
        # Visual feedback on frame
        status_color = (0, 255, 0) if gesture_active else (0, 0, 255)
        if "CLICK" in status_msg or "WAIT" in status_msg: status_color = (255, 0, 255)
        if "DRAG" in status_msg: status_color = (255, 255, 0) # Cyan for drag
        
        cv2.putText(frame, status_msg, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw active zone (margin) with asymmetric top to help reachability
        cv2.rectangle(frame, (int(w*0.1), int(h*0.2)), (int(w*0.9), int(h*0.9)), (255, 255, 255), 1)
        
        cv2.imshow('Gesture Cursor Control', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
