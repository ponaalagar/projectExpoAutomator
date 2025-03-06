import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import math

# Initialize MediaPipe Face Mesh (includes eye landmarks)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize Video Player
video_path = 'vedio/sandeepvedio.mp4'
video_cap = cv2.VideoCapture(video_path)
audio_path = 'audio/sandeepvedio.mp3'  # Assume the audio is extracted to a separate file

# Initialize Pygame for audio playback
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_path)

# State variables
paused = True
no_face_start_time = None
face_detected_time = None
gaze_away_start_time = None
required_face_detection_time = 1  # seconds
required_continuous_detection_time_after_pause = 1  # seconds
face_detected_after_pause_time = None
gaze_away_threshold_time = 1  # seconds - how long user can look away before pausing

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Head pose indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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

def is_user_looking_at_screen(landmarks, image):
    """Determine if user is looking at the screen based on head pose and eye ratio"""
    # Calculate head pose
    pitch, yaw, roll = get_head_pose(landmarks, image)
    
    # Calculate eye aspect ratios to detect blinks/closed eyes
    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)
    
    # Get iris positions to estimate gaze direction
    left_iris_x = sum(landmarks[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
    right_iris_x = sum(landmarks[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
    
    # Calculate relative positions of irises within eyes
    left_eye_width = abs(landmarks[LEFT_EYE[0]].x - landmarks[LEFT_EYE[3]].x)
    right_eye_width = abs(landmarks[RIGHT_EYE[0]].x - landmarks[RIGHT_EYE[3]].x)
    
    left_iris_rel_pos = (left_iris_x - landmarks[LEFT_EYE[0]].x) / left_eye_width if left_eye_width > 0 else 0.5
    right_iris_rel_pos = (right_iris_x - landmarks[RIGHT_EYE[0]].x) / right_eye_width if right_eye_width > 0 else 0.5
    
    # Check if user is looking at screen based on combined factors
    looking_at_screen = True
    
    # Head pose thresholds (adjust these based on your needs)
    if abs(pitch) > 20 or abs(yaw) > 30 or abs(roll) > 20:
        looking_at_screen = False
    
    # Eye gaze thresholds
    avg_iris_pos = (left_iris_rel_pos + right_iris_rel_pos) / 2
    if avg_iris_pos < 0.25 or avg_iris_pos > 0.75:  # Looking too far left or right
        looking_at_screen = False
    
    # Blink detection
    avg_ear = (left_ear + right_ear) / 2
    if avg_ear < 0.2:  # Eyes closed or nearly closed
        # Short blinks are allowed
        pass
    
    return looking_at_screen

# Start the main processing loop
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened() and video_cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        # Make image writeable again
        image_rgb.flags.writeable = True
        
        # Check if face is detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Record time when face is first detected
            if face_detected_time is None:
                face_detected_time = time.time()
            
            # Check if user has been looking at screen long enough to start/resume video
            looking_at_screen = is_user_looking_at_screen(face_landmarks.landmark, image)
            
            if looking_at_screen:
                # Reset gaze away timer
                gaze_away_start_time = None
                
                # Check if we should start/resume the video
                if time.time() - face_detected_time >= required_face_detection_time:
                    if paused and face_detected_after_pause_time is None:
                        face_detected_after_pause_time = time.time()
                    elif paused and face_detected_after_pause_time and time.time() - face_detected_after_pause_time >= required_continuous_detection_time_after_pause:
                        paused = False
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from the beginning
                        pygame.mixer.music.play()  # Play audio from the beginning
                        print("Resuming Video from Beginning")
                        face_detected_time = None
                        face_detected_after_pause_time = None
            else:
                # Start tracking time when gaze moves away
                if gaze_away_start_time is None:
                    gaze_away_start_time = time.time()
                # If gaze has been away for too long, pause the video
                elif time.time() - gaze_away_start_time >= gaze_away_threshold_time and not paused:
                    paused = True
                    pygame.mixer.music.pause()  # Pause audio
                    print("Pausing Video - User not looking at screen")
                    face_detected_after_pause_time = None
            
            # Reset no_face_start_time when face is detected
            no_face_start_time = None
            
            # Draw landmarks and status on image
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw eyes and irises specifically
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
            # Add status text
            status = "Looking at screen" if looking_at_screen else "Not looking at screen"
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if looking_at_screen else (0, 0, 255), 2)
            
            # Display head pose angles
            pitch, yaw, roll = get_head_pose(face_landmarks.landmark, image)
            cv2.putText(image, f"Pitch: {int(pitch)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(image, f"Yaw: {int(yaw)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(image, f"Roll: {int(roll)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
        else:
            # No face detected
            face_detected_time = None
            face_detected_after_pause_time = None
            gaze_away_start_time = None
            
            if no_face_start_time is None:
                no_face_start_time = time.time()
            elif time.time() - no_face_start_time >= 3:  # Faster timeout when no face is detected at all
                if not paused:
                    paused = True
                    pygame.mixer.music.pause()  # Pause audio
                    print("Pausing Video - No face detected")
            
            # Add status text when no face detected
            cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display the webcam feed
        cv2.imshow('Exam Proctor View', image)
        
        # Control video playback
        if not paused:
            ret, frame = video_cap.read()
            if not ret:
                print("End of Video")
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
                continue
            cv2.imshow('Exam Content', frame)
        else:
            # Show paused message on black screen
            black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_screen, "EXAM PAUSED", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if results.multi_face_landmarks:
                if looking_at_screen:
                    remaining_time = required_continuous_detection_time_after_pause - (time.time() - face_detected_after_pause_time) if face_detected_after_pause_time else required_continuous_detection_time_after_pause
                    if remaining_time > 0:
                        cv2.putText(black_screen, f"Resuming in {remaining_time:.1f}s", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                else:
                    cv2.putText(black_screen, "Please look at the screen", (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            else:
                cv2.putText(black_screen, "Face not detected", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                
            cv2.imshow('Exam Content', black_screen)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

# Release all resources
cap.release()
video_cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()
pygame.quit()