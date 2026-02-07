import cv2
import mediapipe as mp
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

def get_head_pose(landmarks, image_width, image_height):
    """
    Estimate head pitch using MediaPipe landmarks.
    Focusing on the vertical orientation (pitch).
    """
    # 3D points of face landmarks (approximation)
    # 33: Left eye, 263: Right eye, 1: Nose tip, 61: Mouth left, 291: Mouth right, 199: Chin
    face_3d = []
    face_2d = []
    
    # Specific landmarks for pose estimation
    landmark_indices = [33, 263, 1, 61, 291, 199]
    
    for idx in landmark_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * image_width), int(lm.y * image_height)
        
        # 2D coordinates
        face_2d.append([x, y])
        
        # 3D coordinates (z is used as a depth relative to landmark 1)
        face_3d.append([x, y, lm.z])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix
    focal_length = 1 * image_width
    cam_matrix = np.array([ [focal_length, 0, image_height / 2],
                            [0, focal_length, image_width / 2],
                            [0, 0, 1]])
    
    # Distortion matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    # Rotation matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # x: pitch, y: yaw, z: roll
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

def main():
    cap = cv2.VideoCapture(0)
    
    # Path to the skeleton video
    video_path = 'skeleton-banging-shield.mp4'
    skeleton_cap = cv2.VideoCapture(video_path)
    
    # Thresholds
    PITCH_THRESHOLD = -15  # Degrees (looking down)
    TIME_THRESHOLD = 1.5   # Seconds
    
    distraction_start_time = None
    is_alerting = False
    window_open = False
    
    print("Anti-Scroll System Started. Press 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Optimization: To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pitch, yaw, roll = get_head_pose(face_landmarks, w, h)
                
                # Logic for distraction detection
                if pitch < PITCH_THRESHOLD:
                    if distraction_start_time is None:
                        distraction_start_time = time.time()
                    else:
                        elapsed_time = time.time() - distraction_start_time
                        if elapsed_time > TIME_THRESHOLD:
                            if not is_alerting:
                                print("\n!!! OSTRZEŻENIE: SKUP SIĘ NA EKRANIE! PRZESTAŃ SCROLLOWAĆ! !!!")
                                is_alerting = True
                                # Prepare alerting window
                                cv2.namedWindow("ALERT: WRACAJ DO PRACY!", cv2.WINDOW_AUTOSIZE)
                                cv2.setWindowProperty("ALERT: WRACAJ DO PRACY!", cv2.WND_PROP_TOPMOST, 1)
                                window_open = True
                else:
                    distraction_start_time = None
                    if is_alerting:
                        print("Wróciłeś do pracy. Tak trzymaj.")
                        is_alerting = False
                        if window_open:
                            cv2.destroyWindow("ALERT: WRACAJ DO PRACY!")
                            window_open = False
                            # Reset video to beginning for next time
                            skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # If alerting, show the skeleton video
                if is_alerting:
                    ret_vid, frame_vid = skeleton_cap.read()
                    if not ret_vid:
                        # Loop video
                        skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_vid, frame_vid = skeleton_cap.read()
                    
                    if ret_vid:
                        cv2.imshow("ALERT: WRACAJ DO PRACY!", frame_vid)
                
                # Visual Feedback (optional, can be disabled for headless)
                status_color = (0, 0, 255) if is_alerting else (0, 255, 0)
                status_text = "ALARM!" if is_alerting else "OK"
                cv2.putText(image, f"Pitch: {int(pitch)} | Status: {status_text}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
        cv2.imshow('Anti-Scroll Detection', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    skeleton_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
