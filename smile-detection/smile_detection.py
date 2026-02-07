import cv2
import mediapipe as mp
import time
import math
import os

# Get path to assets in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    # FPS variables
    prev_time = 0
    
    # Load reaction images
    happy_img = cv2.imread(os.path.join(SCRIPT_DIR, 'happy.jpg'))
    serious_img = cv2.imread(os.path.join(SCRIPT_DIR, 'seriouscat.jpg'))
    
    # Resize images to a consistent size (e.g., 400x400)
    if happy_img is not None:
        happy_img = cv2.resize(happy_img, (400, 400))
    if serious_img is not None:
        serious_img = cv2.resize(serious_img, (400, 400))
    
    print("Smile Detection Started. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        
        # Draw and Detect
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        is_smiling = False
        smile_ratio = 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Get width of the mouth
                mouth_left = landmarks[61]
                mouth_right = landmarks[291]
                mouth_width = math.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
                
                # Get distance between eyes for normalization
                eye_left = landmarks[33]
                eye_right = landmarks[263]
                eye_dist = math.sqrt((eye_left.x - eye_right.x)**2 + (eye_left.y - eye_right.y)**2)
                
                # Calculate Smile Ratio
                if eye_dist > 0:
                    smile_ratio = mouth_width / eye_dist
                
                # Detect Smile: 
                SMILE_THRESHOLD = 0.56 
                
                if smile_ratio > SMILE_THRESHOLD:
                    is_smiling = True
                
                # Draw mouth corners
                h, w, c = image.shape
                for idx in [61, 291]:
                    cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

        # Show Reaction Image
        if is_smiling:
            if happy_img is not None:
                cv2.imshow('Reaction', happy_img)
        else:
            if serious_img is not None:
                cv2.imshow('Reaction', serious_img)

        # Visual Feedback
        status_text = "SMILE :D" if is_smiling else "NO SMILE :|"
        status_color = (0, 255, 0) if is_smiling else (0, 0, 255)
        
        cv2.putText(image, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.putText(image, f"Ratio: {smile_ratio:.2f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(image, f"FPS: {int(fps)}", (w - 100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Smile Detection', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
