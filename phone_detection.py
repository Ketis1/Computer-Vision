import cv2
from ultralytics import YOLO
import time

def main():
    # Load YOLOv8 model (nano version for speed)
    model = YOLO('yolov8n.pt')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Settings
    is_alerting = False
    PHONE_CLASS_ID = 67  # 'cell phone' in COCO dataset
    CONF_THRESHOLD = 0.3 # Lower threshold for better detection
    
    # FPS Calculation variables
    prev_time = 0
    
    print(f"Phone Detection System Started (Conf: {CONF_THRESHOLD}). Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
            
        # Run YOLOv8 detection
        # results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False) # Standard (imgsz=640)
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=320) # Optimized for CPU
        
        phone_detected = False
        
        # Check detected objects
        for r in results:
            for box in r.boxes:
                # Get class ID
                cls = int(box.cls[0])
                
                # If it's a cell phone
                if cls == PHONE_CLASS_ID:
                    phone_detected = True
                    
                    # Draw bounding box for visual feedback
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    conf = float(box.conf[0])
                    cv2.putText(frame, f"TELEFON {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Alert Logic
        if phone_detected:
            if not is_alerting:
                print("\n!!! WYKRYTO TELEFON NA STOLE! PRZESTAŃ SCROLLOWAĆ! !!!")
                is_alerting = True
        else:
            if is_alerting:
                print("Telefon zniknął. Wróć do pracy.")
                is_alerting = False
        
        # Visual Status & FPS
        status_text = "PHONE DETECTED!" if phone_detected else "NO PHONE"
        status_color = (0, 0, 255) if phone_detected else (0, 255, 0)
        
        # Display Status
        cv2.putText(frame, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLOv8 Phone Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
