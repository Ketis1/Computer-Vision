import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Camera
cap = cv2.VideoCapture(0)

# Settings
SCROLL_THRESHOLD = 0.08  # Sensitivity threshold
SCROLL_MULTIPLIER = 400  # How fast it scrolls
SMOOTHING_FACTOR = 0.6   # For exponential smoothing of movement

# Design Colors (Premium Dark Theme)
COLOR_ACCENT = (255, 100, 0)     # Vibrant Blue/Cyan-ish in BGR
COLOR_NEUTRAL = (100, 100, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_BG_OVERLAY = (20, 20, 20)

prev_tilt_val = 0
is_active = True

def draw_premium_ui(frame, tilt_val, scroll_amount):
    h, w, _ = frame.shape
    
    # Create side panel for status
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, h), COLOR_BG_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw Title
    cv2.putText(frame, "HEAD TILT", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2)
    cv2.putText(frame, "SCROLLING", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2)
    
    # Draw Status Indicator
    status_label = "ACTIVE" if is_active else "PAUSED"
    status_color = (0, 255, 100) if is_active else (100, 100, 255)
    cv2.putText(frame, status_label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Draw Tilt Meter
    meter_x = 100
    meter_y_center = h // 2
    meter_height = 200
    
    # Background of the meter
    cv2.rectangle(frame, (meter_x - 10, meter_y_center - meter_height // 2), 
                  (meter_x + 10, meter_y_center + meter_height // 2), COLOR_NEUTRAL, 1)
    
    # Neutral zone marker
    cv2.line(frame, (meter_x - 15, meter_y_center), (meter_x + 15, meter_y_center), COLOR_TEXT, 1)
    
    # Active indicator (the bar)
    bar_h = int(tilt_val * (meter_height // 2) * 5) # Scale for visibility
    bar_h = np.clip(bar_h, -meter_height // 2, meter_height // 2)
    
    color = COLOR_ACCENT if abs(tilt_val) > SCROLL_THRESHOLD else COLOR_NEUTRAL
    cv2.rectangle(frame, (meter_x - 8, meter_y_center), 
                  (meter_x + 8, meter_y_center - bar_h), color, -1)

    # Display scroll feedback
    if abs(scroll_amount) > 0:
        msg = "Scrolling Up" if scroll_amount > 0 else "Scrolling Down"
        cv2.putText(frame, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

def main():
    global is_active, prev_tilt_val
    
    print("Head-Tilt Scrolling started. Press 'q' to quit, 'p' to pause.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        scroll_amount = 0
        current_tilt = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark 4: Nose Tip
                # Landmark 10: Forehead
                # Landmark 152: Chin
                nose = face_landmarks.landmark[4]
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[152]

                # Calculate relative vertical position of the nose
                # Center point between forehead and chin
                center_y = (forehead.y + chin.y) / 2
                face_height = chin.y - forehead.y
                
                # Tilt value normalized by face height
                # Values will be negative when looking up, positive when looking down
                raw_tilt = (nose.y - center_y) / face_height
                
                # Apply smoothing
                current_tilt = (SMOOTHING_FACTOR * raw_tilt) + ((1 - SMOOTHING_FACTOR) * prev_tilt_val)
                prev_tilt_val = current_tilt
                
                # Determine scrolling
                # Note: nose.y increases as we look DOWN. 
                # If nose.y > center_y + threshold -> looking down -> scroll down (negative)
                # If nose.y < center_y - threshold -> looking up -> scroll up (positive)
                
                if is_active:
                    if current_tilt > SCROLL_THRESHOLD:
                        # Looking down
                        intensity = (current_tilt - SCROLL_THRESHOLD) * SCROLL_MULTIPLIER
                        scroll_amount = -int(intensity)
                        pyautogui.scroll(scroll_amount)
                    elif current_tilt < -SCROLL_THRESHOLD:
                        # Looking up
                        intensity = (abs(current_tilt) - SCROLL_THRESHOLD) * SCROLL_MULTIPLIER
                        scroll_amount = int(intensity)
                        pyautogui.scroll(scroll_amount)

        # Draw UI
        draw_premium_ui(frame, current_tilt, scroll_amount)

        cv2.imshow('Head-Tilt Scrolling', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_active = not is_active

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
