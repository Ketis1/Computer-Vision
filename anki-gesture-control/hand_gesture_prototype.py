import cv2
import mediapipe as mp
import time

def count_fingers(hand_landmarks):
    """
    Counts the number of extended fingers.
    Based on MediaPipe landmarks.
    """
    # Landmark indices for fingertips and PIPs
    # Index: 8 (tip), 6 (PIP)
    # Middle: 12 (tip), 10 (PIP)
    # Ring: 16 (tip), 14 (PIP)
    # Pinky: 20 (tip), 18 (PIP)
    # Thumb: 4 (tip), 2 (MCP/base) - Thumb is lateral
    
    fingers = []
    
    # Thumb (lateral check)
    # We compare x-coordinates. For Right hand (palm facing camera): tip < joint
    # For Left hand: tip > joint. We check handedness if possible, but 
    # simple x-comparison works for one side.
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
        
    # Other 4 fingers (vertical check)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def detect_gesture(fingers, hand_landmarks):
    """
    Returns the gesture name based on finger count and specific fingers.
    Mapping:
    - 1 finger (Index only): AGAIN
    - 2 fingers (Index, Middle): HARD
    - 3 fingers (Index, Middle, Ring): GOOD
    - 4 fingers (Index, Middle, Ring, Pinky): EASY
    - 5 fingers (All): SHOW_ANSWER
    """
    num_fingers = sum(fingers)
    thumb, index, middle, ring, pinky = fingers
    
    if num_fingers == 1 and index == 1:
        return "ONE_FINGER"
    elif num_fingers == 2 and index == 1 and middle == 1:
        return "TWO_FINGERS"
    elif num_fingers == 3 and index == 1 and middle == 1 and ring == 1:
        return "THREE_FINGERS"
    elif num_fingers == 4 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "FOUR_FINGERS"
    elif num_fingers == 5:
        return "FIVE_FINGERS"
        
    return "UNKNOWN"

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    # State variables
    current_stable_gesture = "UNKNOWN"
    stable_gesture_start_time = 0
    last_triggered_gesture = "UNKNOWN"
    
    CONFIRMATION_TIME = 1.5 # seconds
    TRIGGER_COOLDOWN = 1.0 # seconds after trigger before allowing another trigger
    last_trigger_time = 0
    
    print("Hand Gesture Prototype Started.")
    print("Gestures (Finger Count):")
    print(" 1 -> ACTION: AGAIN")
    print(" 2 -> ACTION: HARD")
    print(" 3 -> ACTION: GOOD")
    print(" 4 -> ACTION: EASY")
    print(" 5 -> ACTION: SHOW ANSWER")
    print(f"Hold gesture for {CONFIRMATION_TIME}s to trigger.")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        detected_gesture = "UNKNOWN"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                fingers = count_fingers(hand_landmarks)
                detected_gesture = detect_gesture(fingers, hand_landmarks)
        
        # Confirmation Logic
        if detected_gesture != "UNKNOWN":
            if detected_gesture == current_stable_gesture:
                # Still holding the same gesture
                hold_duration = time.time() - stable_gesture_start_time
                
                # Check if enough time passed and not just triggered
                if hold_duration >= CONFIRMATION_TIME and detected_gesture != last_triggered_gesture:
                    if time.time() - last_trigger_time > TRIGGER_COOLDOWN:
                        action = ""
                        if detected_gesture == "ONE_FINGER":
                            action = "AGAIN"
                        elif detected_gesture == "TWO_FINGERS":
                            action = "HARD"
                        elif detected_gesture == "THREE_FINGERS":
                            action = "GOOD"
                        elif detected_gesture == "FOUR_FINGERS":
                            action = "EASY"
                        elif detected_gesture == "FIVE_FINGERS":
                            action = "SHOW ANSWER"
                        
                        if action:
                            print(f"[ACTION TRIGGERED] {action} (Held for {hold_duration:.1f}s)")
                            last_triggered_gesture = detected_gesture
                            last_trigger_time = time.time()
                
                # Visual feedback for progress
                progress = min(hold_duration / CONFIRMATION_TIME, 1.0)
                if detected_gesture != last_triggered_gesture:
                    cv2.rectangle(frame, (10, 70), (int(10 + 200 * progress), 90), (0, 255, 0), -1)
                    cv2.rectangle(frame, (10, 70), (210, 90), (255, 255, 255), 2)
            else:
                # Gesture changed or just started
                current_stable_gesture = detected_gesture
                stable_gesture_start_time = time.time()
                if detected_gesture != last_triggered_gesture:
                     last_triggered_gesture = "UNKNOWN" # Reset trigger memory if it's a new gesture
        else:
            current_stable_gesture = "UNKNOWN"
            last_triggered_gesture = "UNKNOWN"

        status_text = f"Gesture: {detected_gesture}"
        if last_triggered_gesture != "UNKNOWN" and detected_gesture == last_triggered_gesture:
            status_text += " (TRIGGERED)"

        cv2.putText(frame, status_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Gesture Prototype', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
