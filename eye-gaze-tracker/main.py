import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
from utils import calculate_gaze_offset, normalize_offset
from calibration import Calibrator

# Disable PyAutoGUI failsafe to prevent accidental crashes during eye control
pyautogui.FAILSAFE = False

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.calibrator = Calibrator()
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x, self.smooth_y = self.screen_w // 2, self.screen_h // 2
        self.ema_alpha = 0.15 # Smoothing factor
        self.is_moving_mouse = False

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Check if we need calibration
        if not self.calibrator.load_calibration():
            print("Starting calibration...")
            if not self.calibrator.run_calibration(self.face_mesh, cap):
                print("Calibration cancelled.")
                return
        
        print("Gaze tracking active.")
        print("Hold 'M' to move cursor.")
        print("Press 'C' to recalibrate, 'ESC' to quit.")
        
        # Simple window for control and feedback
        cv2.namedWindow("Gaze Control")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            # Check keyboard state
            self.is_moving_mouse = keyboard.is_pressed('m')
            
            # Feedback image
            display_img = image.copy()
            ih, iw, _ = display_img.shape
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                offset = calculate_gaze_offset(landmarks)
                norm_offset = normalize_offset(offset, landmarks)
                
                # Draw small dots on eyes for visual feedback of tracking
                for idx in [468, 473]: # Left and Right iris centers
                    lm = landmarks[idx]
                    cv2.circle(display_img, (int(lm.x * iw), int(lm.y * ih)), 2, (0, 255, 0), -1)

                screen_pos = self.calibrator.transform(norm_offset)
                
                if screen_pos:
                    curr_x, curr_y = screen_pos
                    
                    # Apply Exponential Moving Average (EMA) for smoothing
                    self.smooth_x = self.ema_alpha * curr_x + (1 - self.ema_alpha) * self.smooth_x
                    self.smooth_y = self.ema_alpha * curr_y + (1 - self.ema_alpha) * self.smooth_y
                    
                    # Move cursor if 'M' is held
                    if self.is_moving_mouse:
                        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))
                        status_color = (0, 255, 0)
                        status_text = "ACTIVE"
                    else:
                        status_color = (0, 0, 255)
                        status_text = "INACTIVE (Hold M)"
                    
                    cv2.putText(display_img, f"Status: {status_text}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Show a small preview window
            cv2.imshow('Gaze Control', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break
            elif key == ord('c'): # Recalibrate
                self.calibrator.run_calibration(self.face_mesh, cap)
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GazeTracker()
    tracker.run()
