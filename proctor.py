import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import math
from collections import deque

# Initialize MediaPipe Face Mesh (includes eye landmarks)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize Audio for warnings instead of video
pygame.init()
pygame.mixer.init()

# Sound effects for warnings (replace with actual file paths)
warning_sound_path = 'audio/warning.mp3'  # Create or obtain a warning sound file
try:
    warning_sound = pygame.mixer.Sound(warning_sound_path)
except:
    print("Warning sound not found, creating silent placeholder")
    warning_sound = pygame.mixer.Sound(buffer=np.zeros(44100))  # 1 second of silence

# State variables
exam_active = True
looking_away_counter = 0
consecutive_warnings = 0
max_warnings = 3
warning_cooldown = 10  # seconds
last_warning_time = 0
gaze_history = deque(maxlen=10)  # Track recent gaze directions for smoothing
blink_start_time = None
allowed_blink_duration = 2.0  # Allow eyes to be closed for up to 2 seconds
long_blink_count = 0

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Head pose indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Additional eye corners for more precise gaze tracking
LEFT_EYE_CORNER_OUTER = 263
LEFT_EYE_CORNER_INNER = 362
RIGHT_EYE_CORNER_OUTER = 33
RIGHT_EYE_CORNER_INNER = 133

# Constants for gaze calculation
GAZE_THRESHOLD_X = 0.3  # How far left/right iris can move before considered looking away
GAZE_THRESHOLD_Y = 0.3  # How far up/down iris can move before considered looking away
HEAD_PITCH_THRESHOLD = 15.0  # Degrees
HEAD_YAW_THRESHOLD = 25.0  # Degrees
HEAD_ROLL_THRESHOLD = 15.0  # Degrees
WARNING_THRESHOLD = 15  # Frames of looking away before warning
EYE_CLOSED_THRESHOLD = 0.2  # Threshold for determining if eyes are closed

def get_iris_position(landmarks, iris_indices, eye_corner_inner, eye_corner_outer):
    """Get normalized iris position within the eye"""
    iris_x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)
    iris_y = sum(landmarks[i].y for i in iris_indices) / len(iris_indices)
    
    eye_inner = np.array([landmarks[eye_corner_inner].x, landmarks[eye_corner_inner].y])
    eye_outer = np.array([landmarks[eye_corner_outer].x, landmarks[eye_corner_outer].y])
    
    eye_width = np.linalg.norm(eye_outer - eye_inner)
    eye_center_x = (eye_inner[0] + eye_outer[0]) / 2
    eye_center_y = (eye_inner[1] + eye_outer[1]) / 2
    
    # Normalize iris position relative to eye center and width
    rel_x = (iris_x - eye_center_x) / (eye_width / 2) if eye_width > 0 else 0
    rel_y = (iris_y - eye_center_y) / (eye_width / 3) if eye_width > 0 else 0  # Eye height is approx 2/3 of width
    
    return rel_x, rel_y, iris_x, iris_y

def get_head_pose(landmarks, image):
    """Estimate head pose from face landmarks"""
    image_rows, image_cols, _ = image.shape
    face_3d = []
    face_2d = []
    
    for idx in FACE_OVAL:
        # Convert normalized coordinates to pixel coordinates
        x, y = int(landmarks[idx].x * image_cols), int(landmarks[idx].y * image_rows)
        # Add points to 2D and 3D lists (Z is just an approximation for 3D)
        face_2d.append([x, y])
        face_3d.append([x, y, 0])
    
    # Convert to numpy arrays
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix estimation
    focal_length = image_cols
    camera_center = (image_cols / 2, image_rows / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Find rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d, face_2d, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get angles
    angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))))[6]
    
    # Convert to degrees
    x_angle = angles[0][0]  # pitch
    y_angle = angles[1][0]  # yaw
    z_angle = angles[2][0]  # roll
    
    return x_angle, y_angle, z_angle

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """Calculate the ratio of the vertical to horizontal eye opening"""
    # Vertical eye landmarks
    v1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    v2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    # Horizontal eye landmarks
    h1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    h2 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    
    # Calculate distances
    vertical_dist = np.linalg.norm(v1 - v2)
    horizontal_dist = np.linalg.norm(h1 - h2)
    
    # Return eye aspect ratio
    if horizontal_dist == 0:
        return 0
    return vertical_dist / horizontal_dist

def analyze_gaze(landmarks, image):
    """Analyze gaze direction and head pose to determine user attention"""
    # Get head pose
    pitch, yaw, roll = get_head_pose(landmarks, image)
    
    # Get iris positions
    left_rel_x, left_rel_y, left_x, left_y = get_iris_position(
        landmarks, LEFT_IRIS, LEFT_EYE_CORNER_INNER, LEFT_EYE_CORNER_OUTER)
    right_rel_x, right_rel_y, right_x, right_y = get_iris_position(
        landmarks, RIGHT_IRIS, RIGHT_EYE_CORNER_INNER, RIGHT_EYE_CORNER_OUTER)
    
    # Calculate eye aspect ratios
    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2
    
    # Average the gaze direction from both eyes
    avg_x = (left_rel_x + right_rel_x) / 2
    avg_y = (left_rel_y + right_rel_y) / 2
    
    # Determine if user is looking at screen
    looking_at_screen = True
    look_direction = "Center"
    eyes_closed = False
    
    # Check if eyes are closed
    if avg_ear < EYE_CLOSED_THRESHOLD:
        eyes_closed = True
        look_direction = "Eyes Closed"
        # We don't immediately set looking_at_screen to False
        # This will be handled in the main loop with blink timing
    
    # Only check gaze direction if eyes are open
    elif abs(avg_x) > GAZE_THRESHOLD_X:
        looking_at_screen = False
        look_direction = "Left" if avg_x < 0 else "Right"
    
    # Check vertical gaze
    elif abs(avg_y) > GAZE_THRESHOLD_Y:
        looking_at_screen = False
        look_direction = "Up" if avg_y < 0 else "Down"
    
    # Check head pose
    elif abs(pitch) > HEAD_PITCH_THRESHOLD:
        looking_at_screen = False
        look_direction = "Head Up" if pitch < 0 else "Head Down"
    elif abs(yaw) > HEAD_YAW_THRESHOLD:
        looking_at_screen = False
        look_direction = "Head Left" if yaw < 0 else "Head Right"
    elif abs(roll) > HEAD_ROLL_THRESHOLD:
        looking_at_screen = False
        look_direction = "Head Tilted"
    
    # Calculate precise gaze coordinates (for visualization)
    gaze_data = {
        "left_iris": (left_x, left_y),
        "right_iris": (right_x, right_y),
        "left_rel": (left_rel_x, left_rel_y),
        "right_rel": (right_rel_x, right_rel_y),
        "avg_rel": (avg_x, avg_y),
        "head_pose": (pitch, yaw, roll),
        "looking_at_screen": looking_at_screen,
        "look_direction": look_direction,
        "ear": avg_ear,
        "eyes_closed": eyes_closed
    }
    
    return gaze_data

def issue_warning(image, warning_text, severity=1):
    """Display warning on screen and play sound if needed"""
    global last_warning_time, consecutive_warnings
    
    current_time = time.time()
    if current_time - last_warning_time > warning_cooldown:
        # Reset consecutive warnings after cooldown period
        consecutive_warnings = 0
    
    # Increase warning count
    consecutive_warnings += 1
    last_warning_time = current_time
    
    # Play warning sound
    if consecutive_warnings <= max_warnings:
        warning_sound.play()
    
    # Visual warning on the proctor screen
    overlay = image.copy()
    if severity == 1:  # Mild warning
        color = (0, 255, 255)  # Yellow
    elif severity == 2:  # Moderate warning
        color = (0, 165, 255)  # Orange
    else:  # Severe warning
        color = (0, 0, 255)  # Red
    
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), color, -1)
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add warning text
    cv2.putText(image, warning_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"Warning {consecutive_warnings}/{max_warnings}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if consecutive_warnings > max_warnings:
        cv2.putText(image, "EXAM VIOLATION DETECTED", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return image, consecutive_warnings > max_warnings

def render_gaze_visualization(image, gaze_data, landmarks):
    """Render visualization of gaze tracking"""
    h, w = image.shape[:2]
    
    # Draw eye contours
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        points = []
        for i in eye_idx:
            x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
            points.append((x, y))
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 1)
    
    # Draw irises if eyes are open
    if not gaze_data["eyes_closed"]:
        for iris_idx, color in zip([LEFT_IRIS, RIGHT_IRIS], [(255, 0, 0), (0, 0, 255)]):
            points = []
            for i in iris_idx:
                x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
                points.append((x, y))
            center = np.mean(points, axis=0).astype(np.int32)
            cv2.circle(image, tuple(center), 2, color, -1)
        
        # Add gaze direction visualization
        left_eye_center = np.array(gaze_data["left_iris"]) * np.array([w, h])
        right_eye_center = np.array(gaze_data["right_iris"]) * np.array([w, h])
        
        # Draw gaze direction lines
        left_gaze = gaze_data["left_rel"]
        right_gaze = gaze_data["right_rel"]
        
        left_gaze_point = (int(left_eye_center[0] + left_gaze[0] * 50), 
                          int(left_eye_center[1] + left_gaze[1] * 50))
        right_gaze_point = (int(right_eye_center[0] + right_gaze[0] * 50), 
                           int(right_eye_center[1] + right_gaze[1] * 50))
        
        cv2.line(image, (int(left_eye_center[0]), int(left_eye_center[1])), left_gaze_point, (255, 0, 0), 2)
        cv2.line(image, (int(right_eye_center[0]), int(right_eye_center[1])), right_gaze_point, (0, 0, 255), 2)
    
    # Add status information
    pitch, yaw, roll = gaze_data["head_pose"]
    
    # Determine status color - special case for closed eyes within allowed time
    if gaze_data["eyes_closed"] and blink_start_time and (time.time() - blink_start_time <= allowed_blink_duration):
        status_color = (0, 255, 255)  # Yellow for normal blink/eye closure
    else:
        status_color = (0, 255, 0) if gaze_data["looking_at_screen"] else (0, 0, 255)
    
    cv2.putText(image, f"Gaze: {gaze_data['look_direction']}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Add blink timing if eyes are closed
    if gaze_data["eyes_closed"] and blink_start_time:
        blink_duration = time.time() - blink_start_time
        if blink_duration <= allowed_blink_duration:
            cv2.putText(image, f"Blink: {blink_duration:.1f}s/{allowed_blink_duration:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(image, f"Eyes closed too long: {blink_duration:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(image, f"Pitch: {int(pitch)}", (10, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Yaw: {int(yaw)}", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Roll: {int(roll)}", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"EAR: {gaze_data['ear']:.2f}", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return image

# Main processing loop
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened() and exam_active:
        success, image = cap.read()
        if not success:
            print("Failed to read webcam")
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        # Make image writeable again
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        violation_detected = False
        
        # Check if face is detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Analyze gaze and head position
            gaze_data = analyze_gaze(face_landmarks.landmark, image)
            
            # Handle eye closure separately
            if gaze_data["eyes_closed"]:
                if blink_start_time is None:
                    blink_start_time = time.time()
                    
                blink_duration = time.time() - blink_start_time
                
                # Only treat as not looking at screen if eyes closed too long
                if blink_duration > allowed_blink_duration:
                    gaze_data["looking_at_screen"] = False
                    
                    # Count excessive eye closures
                    if looking_away_counter == 0:  # Only increment once per long closure
                        long_blink_count += 1
                        
                    looking_away_counter += 1
                    
                    # Issue warning for excessive eye closure
                    if looking_away_counter >= WARNING_THRESHOLD:
                        warning_text = f"Eyes closed for {blink_duration:.1f}s (limit: {allowed_blink_duration}s)"
                        
                        # Determine warning severity
                        severity = 1
                        if looking_away_counter >= WARNING_THRESHOLD * 2:
                            severity = 2
                        if looking_away_counter >= WARNING_THRESHOLD * 3 or long_blink_count > 5:
                            severity = 3
                        
                        image, violation_detected = issue_warning(image, warning_text, severity)
                else:
                    # Eyes are closed but within allowed duration - this is fine
                    looking_away_counter = 0
                    gaze_history.append(True)  # Count as looking at screen
            else:
                # Eyes are open, reset blink timer
                blink_start_time = None
                
                # Add current gaze status to history
                gaze_history.append(gaze_data["looking_at_screen"])
                
                # Check for looking away (but not due to closed eyes)
                if not gaze_data["looking_at_screen"]:
                    looking_away_counter += 1
                    
                    # Issue warnings if looking away for too long
                    if looking_away_counter >= WARNING_THRESHOLD:
                        warning_text = f"Eyes detected looking {gaze_data['look_direction']}"
                        
                        # Determine warning severity
                        severity = 1
                        if looking_away_counter >= WARNING_THRESHOLD * 2:
                            severity = 2
                        if looking_away_counter >= WARNING_THRESHOLD * 3:
                            severity = 3
                        
                        image, violation_detected = issue_warning(image, warning_text, severity)
                else:
                    looking_away_counter = max(0, looking_away_counter - 1)  # Gradually reduce counter when looking at screen
            
            # Render visualization
            image = render_gaze_visualization(image, gaze_data, face_landmarks.landmark)
            
        else:
            # No face detected
            blink_start_time = None  # Reset blink timer when no face detected
            looking_away_counter += 2  # Increase faster when no face detected
            
            if looking_away_counter >= WARNING_THRESHOLD:
                image, violation_detected = issue_warning(image, "Face not detected", 3)
            
            # Add status text when no face detected
            cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display monitoring information
        cv2.putText(image, "Exam Proctoring System", (10, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show blink statistics
        cv2.putText(image, f"Long eye closures: {long_blink_count}", 
                   (image.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Handle severe violations
        if violation_detected:
            cv2.putText(image, "EXAM INTEGRITY COMPROMISED", 
                        (image.shape[1]//2 - 200, image.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # You could implement automatic reporting or exam termination here
        
        # Display the proctor monitoring view
        cv2.imshow('Exam Proctor View', image)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release all resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()