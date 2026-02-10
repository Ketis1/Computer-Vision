import numpy as np

# MediaPipe Landmark Indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def get_eye_center(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    return np.mean(points, axis=0)

def get_iris_center(landmarks, iris_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in iris_indices])
    return np.mean(points, axis=0)

def calculate_gaze_offset(landmarks):
    """
    Calculates the relative offset of the iris within the eye.
    Returns a unified offset (x, y) based on both eyes.
    """
    left_eye_center = get_eye_center(landmarks, LEFT_EYE)
    left_iris_center = get_iris_center(landmarks, LEFT_IRIS)
    
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE)
    right_iris_center = get_iris_center(landmarks, RIGHT_IRIS)
    
    # Calculate offset
    left_offset = left_iris_center - left_eye_center
    right_offset = right_iris_center - right_eye_center
    
    # Average offset from both eyes
    avg_offset = (left_offset + right_offset) / 2.0
    
    return avg_offset

def normalize_offset(offset, landmarks):
    """
    Normalizes the offset by the eye width to handle distance from camera.
    """
    # Use distance between inner and outer eye corners as scale
    left_width = np.linalg.norm(
        np.array([landmarks[362].x - landmarks[263].x, landmarks[362].y - landmarks[263].y])
    )
    right_width = np.linalg.norm(
        np.array([landmarks[33].x - landmarks[133].x, landmarks[33].y - landmarks[133].y])
    )
    avg_width = (left_width + right_width) / 2.0
    
    return offset / avg_width
