import os
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
from collections import deque
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

# Session state
class ProctorSessionState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.exam_active = True
        self.looking_away_counter = 0
        self.consecutive_warnings = 0
        self.max_warnings = 3
        self.warning_cooldown = 10  # seconds
        self.last_warning_time = 0
        self.gaze_history = deque(maxlen=10)  # Track recent gaze directions for smoothing
        self.blink_start_time = None
        self.allowed_blink_duration = 2.0  # Allow eyes to be closed for up to 2 seconds
        self.long_blink_count = 0
        self.violations = []
        self.current_gaze_data = None
        self.face_detected = False
        self.last_activity = time.time()

# Store active sessions
sessions = {}

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

def get_head_pose(landmarks, image_shape):
    """Estimate head pose from face landmarks"""
    image_rows, image_cols = image_shape[:2]
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

def analyze_gaze(landmarks, image_shape):
    """Analyze gaze direction and head pose to determine user attention"""
    # Get head pose
    pitch, yaw, roll = get_head_pose(landmarks, image_shape)
    
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
        # This will be handled with blink timing
    
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

def issue_warning(session, warning_text, severity=1):
    """Record a warning in the session"""
    current_time = time.time()
    if current_time - session.last_warning_time > session.warning_cooldown:
        # Reset consecutive warnings after cooldown period
        session.consecutive_warnings = 0
    
    # Increase warning count
    session.consecutive_warnings += 1
    session.last_warning_time = current_time
    
    # Record the violation
    violation = {
        "timestamp": current_time,
        "text": warning_text,
        "severity": severity,
        "warning_number": session.consecutive_warnings,
        "is_major": session.consecutive_warnings > session.max_warnings
    }
    
    session.violations.append(violation)
    return violation

def process_frame(session_id, frame_data):
    """Process a webcam frame for proctoring"""
    if session_id not in sessions:
        sessions[session_id] = ProctorSessionState(session_id)
        
    session = sessions[session_id]
    session.last_activity = time.time()
    
    # Decode base64 image
    try:
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return {
            "error": "Failed to decode image data",
            "status": "error"
        }
    
    # Process the frame with MediaPipe
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        
        violations = []
        violation_detected = False
        session.face_detected = False
        
        # Check if face is detected
        if results.multi_face_landmarks:
            session.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Analyze gaze and head position
            gaze_data = analyze_gaze(face_landmarks.landmark, frame.shape)
            session.current_gaze_data = gaze_data
            
            # Handle eye closure separately
            if gaze_data["eyes_closed"]:
                if session.blink_start_time is None:
                    session.blink_start_time = time.time()
                    
                blink_duration = time.time() - session.blink_start_time
                
                # Only treat as not looking at screen if eyes closed too long
                if blink_duration > session.allowed_blink_duration:
                    gaze_data["looking_at_screen"] = False
                    
                    # Count excessive eye closures
                    if session.looking_away_counter == 0:  # Only increment once per long closure
                        session.long_blink_count += 1
                        
                    session.looking_away_counter += 1
                    
                    # Issue warning for excessive eye closure
                    if session.looking_away_counter >= WARNING_THRESHOLD:
                        warning_text = f"Eyes closed for {blink_duration:.1f}s (limit: {session.allowed_blink_duration}s)"
                        
                        # Determine warning severity
                        severity = 1
                        if session.looking_away_counter >= WARNING_THRESHOLD * 2:
                            severity = 2
                        if session.looking_away_counter >= WARNING_THRESHOLD * 3 or session.long_blink_count > 5:
                            severity = 3
                        
                        violation = issue_warning(session, warning_text, severity)
                        violations.append(violation)
                        if violation["is_major"]:
                            violation_detected = True
                else:
                    # Eyes are closed but within allowed duration - this is fine
                    session.looking_away_counter = 0
                    session.gaze_history.append(True)  # Count as looking at screen
            else:
                # Eyes are open, reset blink timer
                session.blink_start_time = None
                
                # Add current gaze status to history
                session.gaze_history.append(gaze_data["looking_at_screen"])
                
                # Check for looking away (but not due to closed eyes)
                if not gaze_data["looking_at_screen"]:
                    session.looking_away_counter += 1
                    
                    # Issue warnings if looking away for too long
                    if session.looking_away_counter >= WARNING_THRESHOLD:
                        warning_text = f"Eyes detected looking {gaze_data['look_direction']}"
                        
                        # Determine warning severity
                        severity = 1
                        if session.looking_away_counter >= WARNING_THRESHOLD * 2:
                            severity = 2
                        if session.looking_away_counter >= WARNING_THRESHOLD * 3:
                            severity = 3
                        
                        violation = issue_warning(session, warning_text, severity)
                        violations.append(violation)
                        if violation["is_major"]:
                            violation_detected = True
                else:
                    session.looking_away_counter = max(0, session.looking_away_counter - 1)  # Gradually reduce counter when looking at screen
        else:
            # No face detected
            session.blink_start_time = None  # Reset blink timer when no face detected
            session.looking_away_counter += 2  # Increase faster when no face detected
            
            if session.looking_away_counter >= WARNING_THRESHOLD:
                violation = issue_warning(session, "Face not detected", 3)
                violations.append(violation)
                if violation["is_major"]:
                    violation_detected = True
    
    # Prepare response
    response = {
        "status": "violation" if violation_detected else "ok",
        "faceDetected": session.face_detected,
        "lookingAway": session.looking_away_counter > 0,
        "lookingAwayCounter": session.looking_away_counter,
        "longBlinkCount": session.long_blink_count,
        "warningCount": session.consecutive_warnings,
        "maxWarnings": session.max_warnings,
        "currentViolations": violations,
        "gazeDirection": session.current_gaze_data["look_direction"] if session.current_gaze_data else "Unknown",
        "eyesClosed": session.current_gaze_data["eyes_closed"] if session.current_gaze_data else False
    }
    
    return response

@app.route('/api/proctor/start', methods=['POST'])
def start_session():
    """Start a new proctoring session"""
    data = request.json
    session_id = data.get('sessionId', str(time.time()))
    
    if session_id in sessions:
        # Reset existing session
        sessions[session_id] = ProctorSessionState(session_id)
    else:
        # Create new session
        sessions[session_id] = ProctorSessionState(session_id)
    
    return jsonify({
        "status": "success",
        "sessionId": session_id,
        "message": "Proctoring session started"
    })

@app.route('/api/proctor/process', methods=['POST'])
def process_frame_api():
    """Process a single frame from the webcam"""
    data = request.json
    session_id = data.get('sessionId')
    frame_data = data.get('frameData')
    
    if not session_id or not frame_data:
        return jsonify({
            "status": "error",
            "message": "Missing sessionId or frameData"
        }), 400
    
    result = process_frame(session_id, frame_data)
    return jsonify(result)

@app.route('/api/proctor/status', methods=['GET'])
def get_session_status():
    """Get the current status of a proctoring session"""
    session_id = request.args.get('sessionId')
    
    if not session_id or session_id not in sessions:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing sessionId"
        }), 400
    
    session = sessions[session_id]
    
    # Summarize violations
    violation_summary = {
        "total": len(session.violations),
        "major": sum(1 for v in session.violations if v.get("is_major", False)),
        "minor": sum(1 for v in session.violations if not v.get("is_major", False)),
        "recent": [v for v in session.violations[-5:]] if session.violations else []
    }
    
    return jsonify({
        "status": "active" if session.exam_active else "inactive",
        "faceDetected": session.face_detected,
        "lookingAway": session.looking_away_counter > 0,
        "consecutiveWarnings": session.consecutive_warnings,
        "maxWarningsAllowed": session.max_warnings,
        "longBlinkCount": session.long_blink_count,
        "violations": violation_summary,
        "lastActivity": session.last_activity
    })

@app.route('/api/proctor/end', methods=['POST'])
def end_session():
    """End a proctoring session and get final results"""
    data = request.json
    session_id = data.get('sessionId')
    
    if not session_id or session_id not in sessions:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing sessionId"
        }), 400
    
    session = sessions[session_id]
    session.exam_active = False
    
    # Generate session report
    violation_count = len(session.violations)
    major_violations = sum(1 for v in session.violations if v.get("is_major", False))
    
    # Determine if exam was compromised
    exam_compromised = major_violations > 0
    
    # Create session report
    report = {
        "sessionId": session_id,
        "duration": time.time() - session.last_activity,
        "totalViolations": violation_count,
        "majorViolations": major_violations,
        "minorViolations": violation_count - major_violations,
        "longEyeClosures": session.long_blink_count,
        "examCompromised": exam_compromised,
        "violationDetails": session.violations
    }
    
    # Cleanup session data (optional, can keep for record)
    # del sessions[session_id]
    
    return jsonify({
        "status": "success",
        "message": "Proctoring session ended",
        "report": report
    })

# Cleanup thread to remove inactive sessions
def cleanup_inactive_sessions():
    while True:
        try:
            current_time = time.time()
            inactive_sessions = []
            
            for session_id, session in sessions.items():
                # If session has been inactive for 30 minutes
                if current_time - session.last_activity > 1800:
                    inactive_sessions.append(session_id)
            
            # Remove inactive sessions
            for session_id in inactive_sessions:
                del sessions[session_id]
                logger.info(f"Removed inactive session: {session_id}")
            
            # Sleep for 5 minutes
            time.sleep(300)
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
