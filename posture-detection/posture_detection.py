import cv2
import mediapipe as mp
import time
import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    # FPS variables
    prev_time = 0
    
    # Posture variables
    is_slouching = False
    
    # Calibration variables
    calibrated_neck_dist = None
    
    print("Posture Detection Started.")
    print("1. Sit straight and press 'c' to calibrate.")
    print("2. System will alert when you slouch relative to your calibrated pose.")
    print("3. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process landmarks
        results = pose.process(image)
        
        # Draw and detect posture
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        current_neck_dist = 0
        
        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Extract landmarks for posture analysis
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Points for posture
                l_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                r_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                l_ear_y = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y
                r_ear_y = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y

                # Vertical distance between ear and shoulder
                avg_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
                avg_ear_y = (l_ear_y + r_ear_y) / 2
                current_neck_dist = avg_shoulder_y - avg_ear_y

                # Detection logic based on calibration
                if calibrated_neck_dist is not None:
                    # If current distance is less than 85% of calibrated distance -> slouching
                    if current_neck_dist < (calibrated_neck_dist * 0.85):
                        is_slouching = True
                    else:
                        is_slouching = False
                
            except Exception as e:
                pass

        # Visual Feedback
        if calibrated_neck_dist is None:
            status_text = "SIT STRAIGHT AND PRESS 'C'"
            status_color = (0, 255, 255) # Yellowish
        else:
            status_text = "STRAIGHTEN UP!" if is_slouching else "POSTURE OK"
            status_color = (0, 0, 255) if is_slouching else (0, 255, 0)
        
        cv2.putText(image, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.putText(image, f"FPS: {int(fps)}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if calibrated_neck_dist is not None:
             cv2.putText(image, "CALIBRATED", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show frame
        cv2.imshow('Posture Detection (MediaPipe)', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if results.pose_landmarks:
                calibrated_neck_dist = current_neck_dist
                print(f"Calibrated! Ideal distance: {calibrated_neck_dist:.4f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
