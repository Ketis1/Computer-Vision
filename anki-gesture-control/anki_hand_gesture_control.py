import cv2
import mediapipe as mp
import time
import requests
import json

# AnkiConnect configuration
ANKI_CONNECT_URL = 'http://127.0.0.1:8765'

def invoke_anki(action, **params):
    """
    Sends a request to AnkiConnect.
    """
    try:
        payload = json.dumps({'action': action, 'version': 6, 'params': params})
        response = requests.post(ANKI_CONNECT_URL, data=payload, timeout=2).json()
        if response.get('error'):
            print(f"[Anki Error] {response['error']}")
            return None
        return response.get('result')
    except Exception as e:
        print(f"[Connection Error] Could not connect to Anki: {e}")
        return None

def count_fingers(hand_landmarks):
    """
    Counts which fingers are extended.
    Returns: [thumb, index, middle, ring, pinky]
    """
    fingers = []
    
    # Thumb (lateral check) - Simplified for mirror view (right hand on screen is left hand in reality)
    # MediaPipe landmarks for thumb: 4 (tip), 3 (joint)
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

def detect_gesture(fingers):
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
        return "AGAIN"
    elif num_fingers == 2 and index == 1 and middle == 1:
        return "HARD"
    elif num_fingers == 3 and index == 1 and middle == 1 and ring == 1:
        return "GOOD"
    elif num_fingers == 4 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "EASY"
    elif num_fingers == 5:
        return "SHOW_ANSWER"
        
    return "UNKNOWN"

def perform_anki_action(gesture):
    """
    Triggers the corresponding action in Anki.
    """
    if gesture == "SHOW_ANSWER":
        print("[Anki] Showing Answer...")
        invoke_anki('guiShowAnswer')
    elif gesture == "AGAIN":
        print("[Anki] Rating: AGAIN")
        invoke_anki('guiAnswerCard', ease=1)
    elif gesture == "HARD":
        print("[Anki] Rating: HARD")
        invoke_anki('guiAnswerCard', ease=2)
    elif gesture == "GOOD":
        print("[Anki] Rating: GOOD")
        invoke_anki('guiAnswerCard', ease=3)
    elif gesture == "EASY":
        print("[Anki] Rating: EASY")
        invoke_anki('guiAnswerCard', ease=4)

def main():
    # Initialize MediaPipe
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
    TRIGGER_COOLDOWN = 1.5 # seconds cooldown
    last_trigger_time = 0
    
    print("Anki Hand Gesture Control Started.")
    print("Controls:")
    print(" 1 Finger (Index) -> AGAIN")
    print(" 2 Fingers        -> HARD")
    print(" 3 Fingers        -> GOOD")
    print(" 4 Fingers        -> EASY")
    print(" 5 Fingers        -> SHOW ANSWER")
    print(f"Hold gesture for {CONFIRMATION_TIME}s to trigger.")
    print("Press 'q' to quit.")
    
    # Check Anki connection initially
    if invoke_anki('version'):
        print("[System] Successfully connected to Anki Desktop.")
    else:
        print("[System] WARNING: Could not connect to Anki Desktop. Make sure Anki and AnkiConnect are running.")

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
                detected_gesture = detect_gesture(fingers)
        
        # Confirmation Logic
        if detected_gesture != "UNKNOWN":
            if detected_gesture == current_stable_gesture:
                hold_duration = time.time() - stable_gesture_start_time
                
                # Check for triggering
                if hold_duration >= CONFIRMATION_TIME and detected_gesture != last_triggered_gesture:
                    if time.time() - last_trigger_time > TRIGGER_COOLDOWN:
                        print(f"[Trigger] Gesture confirmed: {detected_gesture}")
                        perform_anki_action(detected_gesture)
                        last_triggered_gesture = detected_gesture
                        last_trigger_time = time.time()
                
                # Visual feedback
                progress = min(hold_duration / CONFIRMATION_TIME, 1.0)
                if detected_gesture != last_triggered_gesture:
                    cv2.rectangle(frame, (10, 70), (int(10 + 200 * progress), 90), (0, 255, 0), -1)
                    cv2.rectangle(frame, (10, 70), (210, 90), (255, 255, 255), 2)
            else:
                current_stable_gesture = detected_gesture
                stable_gesture_start_time = time.time()
                if detected_gesture != last_triggered_gesture:
                    last_triggered_gesture = "UNKNOWN"
        else:
            current_stable_gesture = "UNKNOWN"
            # We don't reset last_triggered_gesture here to allow hands to leave and come back
            # or we can reset it if we want to allow the same gesture again immediately.
            # Let's reset it to allow re-triggering easily if hands are removed.
            last_triggered_gesture = "UNKNOWN"

        # Overlay text
        status_text = f"Anki Gesture: {detected_gesture}"
        cv2.putText(frame, status_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Anki Hand Gesture Control', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
