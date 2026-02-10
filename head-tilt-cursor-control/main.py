import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import time

# Disable PyAutoGUI failsafe for smooth movement
pyautogui.FAILSAFE = False

class HeadJoystick:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Calibration / Neutral position
        self.neutral_pitch = 0
        self.neutral_yaw = 0
        self.is_calibrated = False
        
        # Settings
        self.deadzone_x = 20.0 
        self.deadzone_y = 4.0 
        self.speed_multiplier_x = 3.0 
        self.speed_multiplier_y = 10.0 
        
        # Smoothing for movement
        self.smooth_dx = 0
        self.smooth_dy = 0
        self.ema_alpha = 0.2
        
        # Click detection (Blink)
        self.left_eye_closed_start = None
        self.click_threshold = 1.0 # 1 second hold for click
        self.ear_threshold = 0.22  # EAR threshold for closed eye
        
        self.screen_h, self.screen_w = pyautogui.size()
        self.is_active = False

    def get_head_pose(self, landmarks, img_w, img_h):
        # 3D model points (standard facial features)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        # 2D image points from MediaPipe (indices: nose=1, chin=152, L eye=33, R eye=263, L mouth=61, R mouth=291)
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[33].x * img_w, landmarks[33].y * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[61].x * img_w, landmarks[61].y * img_h),
            (landmarks[291].x * img_w, landmarks[291].y * img_h)
        ], dtype=np.float32)

        # Camera internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float32
        )

        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        # decomposeProjectionMatrix returns 7 values: cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles
        projection_matrix = np.hstack((rmat, translation_vector))
        res = cv2.decomposeProjectionMatrix(projection_matrix)
        angles = res[6] # Euler angles are the 7th element
        
        # angles contains [Pitch, Yaw, Roll]
        return angles[0].item(), angles[1].item()

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Head Tilt Joystick Active.")
        print("Press 'C' to set neutral position.")
        print("Hold 'RIGHT CTRL' to move cursor.")
        print("Press 'ESC' to quit.")

        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            display_img = image.copy()
            active_color = (0, 0, 255) # Red by default

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pitch, yaw = self.get_head_pose(landmarks, w, h)
                
                # Calibration
                if keyboard.is_pressed('c'):
                    self.neutral_pitch = pitch
                    self.neutral_yaw = yaw
                    self.is_calibrated = True
                    print(f"Calibrated: Neutral Pitch={pitch:.2f}, Yaw={yaw:.2f}")

                if self.is_calibrated:
                    # Calculate relative tilt with wrap-around normalization
                    rel_pitch = pitch - self.neutral_pitch
                    rel_yaw = yaw - self.neutral_yaw
                    
                    # Normalize angles to [-180, 180] to handle the 180/-180 jump
                    rel_pitch = (rel_pitch + 180) % 360 - 180
                    rel_yaw = (rel_yaw + 180) % 360 - 180
                    
                    # Trackpoint Logic: Determine speed based on tilt
                    dx, dy = 0, 0
                    
                    # Horizontal (Yaw) - INVERTED
                    if abs(rel_yaw) > self.deadzone_x:
                        # Inverting direction: -1 if rel_yaw > 0, 1 otherwise
                        direction = -1 if rel_yaw > 0 else 1
                        dx = direction * (abs(rel_yaw) - self.deadzone_x) * self.speed_multiplier_x
                    
                    # Vertical (Pitch)
                    if abs(rel_pitch) > self.deadzone_y:
                        direction = 1 if rel_pitch > 0 else -1
                        dy = direction * (abs(rel_pitch) - self.deadzone_y) * self.speed_multiplier_y

                    # Apply Smoothing (EMA)
                    self.smooth_dx = self.ema_alpha * dx + (1 - self.ema_alpha) * self.smooth_dx
                    self.smooth_dy = self.ema_alpha * dy + (1 - self.ema_alpha) * self.smooth_dy

                    # Apply movement if 'RIGHT CTRL' is held
                    is_active = keyboard.is_pressed('right ctrl')
                    if is_active:
                        active_color = (0, 255, 0)
                        pyautogui.moveRel(int(self.smooth_dx), int(self.smooth_dy))
                    
                    status_text = "ACTIVE" if is_active else "INACTIVE (Hold R_CTRL)"
                    cv2.putText(display_img, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2)
                    cv2.putText(display_img, f"P: {rel_pitch:.1f} (th:{self.deadzone_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display_img, f"Y: {rel_yaw:.1f} (th:{self.deadzone_x})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # --- Click Detection (Right Eye Blink) ---
                    # Right Eye Indices: 133 (inner), 33 (outer), 159 (top), 145 (bottom)
                    p_inner = landmarks[133]
                    p_outer = landmarks[33]
                    p_top = landmarks[159]
                    p_bottom = landmarks[145]

                    # Horizontal distance
                    h_dist = np.sqrt((p_inner.x - p_outer.x)**2 + (p_inner.y - p_outer.y)**2)
                    # Vertical distance
                    v_dist = np.sqrt((p_top.x - p_bottom.x)**2 + (p_top.y - p_bottom.y)**2)
                    
                    ear = v_dist / h_dist if h_dist > 0 else 0
                    
                    if ear < self.ear_threshold:
                        if self.left_eye_closed_start is None:
                            self.left_eye_closed_start = time.time()
                        
                        hold_duration = time.time() - self.left_eye_closed_start
                        
                        # Visual progress bar for click
                        bar_w = int(min(hold_duration / self.click_threshold, 1.0) * 150)
                        cv2.rectangle(display_img, (10, 110), (160, 125), (100, 100, 100), 2)
                        cv2.rectangle(display_img, (10, 110), (10 + bar_w, 125), (255, 255, 0), -1)
                        cv2.putText(display_img, "CLICKING...", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        if hold_duration >= self.click_threshold:
                            pyautogui.click()
                            print("Left Click Triggered!")
                            self.left_eye_closed_start = time.time() + 1.0 # Prevent multi-clicks
                    else:
                        self.left_eye_closed_start = None

                    # Visual feedback of the "Joystick"
                    center_view = (w // 2, h // 2)
                    cv2.line(display_img, center_view, (int(center_view[0] + rel_yaw*10), int(center_view[1] + rel_pitch*10)), active_color, 3)
                    cv2.circle(display_img, center_view, 20, (255, 255, 255), 1)

            cv2.imshow('Head Tilt Control', display_img)
            if cv2.waitKey(1) & 0xFF == 27: break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    joy = HeadJoystick()
    joy.run()
