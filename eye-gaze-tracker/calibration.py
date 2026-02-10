import cv2
import numpy as np
import json
import os
import pyautogui
from utils import calculate_gaze_offset, normalize_offset

class Calibrator:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        # 9-point grid (3x3) for better coverage
        self.points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.calibration_data = []
        self.map_x = None # Coefficients for X mapping
        self.map_y = None # Coefficients for Y mapping

    def _get_features(self, offset):
        """Creates polynomial features: [1, x, y, x*y, x^2, y^2]"""
        x, y = offset
        return [1, x, y, x*y, x**2, y**2]

    def run_calibration(self, face_mesh, cap):
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.calibration_data = [] # Reset old data
        
        for p_x, p_y in self.points:
            target_x = int(p_x * self.screen_w)
            target_y = int(p_y * self.screen_h)
            
            # Phase 1: Preparation (Giving user time to focus)
            for i in range(2, 0, -1):
                start_time = cv2.getTickCount()
                while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 1.0:
                    success, image = cap.read()
                    display = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    cv2.circle(display, (target_x, target_y), 40, (100, 100, 100), 2)
                    cv2.putText(display, f"Look at the dot in {i}...", (self.screen_w // 2 - 250, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Calibration", display)
                    if cv2.waitKey(1) & 0xFF == 27: return False

            # Phase 2: Collection
            collected_offsets = []
            max_samples = 40 # Slightly fewer samples per point to keep 9-point flow fast
            while len(collected_offsets) < max_samples:
                success, image = cap.read()
                if not success: break
                
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)
                
                display = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                progress = int((len(collected_offsets) / max_samples) * 360)
                cv2.ellipse(display, (target_x, target_y), (30, 30), 0, 0, progress, (0, 255, 0), -1)
                cv2.putText(display, "Calibrating...", (self.screen_w // 2 - 150, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    offset = calculate_gaze_offset(landmarks)
                    norm_offset = normalize_offset(offset, landmarks)
                    collected_offsets.append(norm_offset)
                
                cv2.imshow("Calibration", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    return False
            
            avg_offset = np.mean(collected_offsets, axis=0)
            self.calibration_data.append({
                "screen_point": [target_x, target_y],
                "eye_offset": avg_offset.tolist()
            })
            
        cv2.destroyWindow("Calibration")
        self.train_model()
        self.save_calibration()
        return True

    def train_model(self):
        """Solves for polynomial mapping coefficients using Least Squares"""
        if len(self.calibration_data) < 6: return
        
        A = []
        B_x = []
        B_y = []
        
        for data in self.calibration_data:
            features = self._get_features(data["eye_offset"])
            A.append(features)
            B_x.append(data["screen_point"][0])
            B_y.append(data["screen_point"][1])
            
        A = np.array(A)
        self.map_x, _, _, _ = np.linalg.lstsq(A, np.array(B_x), rcond=None)
        self.map_y, _, _, _ = np.linalg.lstsq(A, np.array(B_y), rcond=None)

    def transform(self, norm_offset):
        if self.map_x is None or self.map_y is None:
            return None
            
        features = np.array(self._get_features(norm_offset))
        pred_x = np.dot(features, self.map_x)
        pred_y = np.dot(features, self.map_y)
        
        # Clamp to screen size
        pred_x = np.clip(pred_x, 0, self.screen_w)
        pred_y = np.clip(pred_y, 0, self.screen_h)
        
        return pred_x, pred_y

    def save_calibration(self):
        data = {
            "calibration_points": self.calibration_data,
            "map_x": self.map_x.tolist() if self.map_x is not None else None,
            "map_y": self.map_y.tolist() if self.map_y is not None else None
        }
        with open("calibration.json", "w") as f:
            json.dump(data, f)

    def load_calibration(self):
        if os.path.exists("calibration.json"):
            with open("calibration.json", "r") as f:
                data = json.load(f)
                self.calibration_data = data["calibration_points"]
                if data["map_x"] and data["map_y"]:
                    self.map_x = np.array(data["map_x"])
                    self.map_y = np.array(data["map_y"])
                    return True
        return False
