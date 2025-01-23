import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

paused = True
no_face_start_time = None
face_detected_time = None
required_face_detection_time = 2  # seconds
required_continuous_detection_time_after_pause = 3  # seconds
face_detected_after_pause_time = None

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened() and video_cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Check if any face is detected
        if results.detections:
            if face_detected_time is None:
                face_detected_time = time.time()
            elif time.time() - face_detected_time >= required_face_detection_time:
                if paused and face_detected_after_pause_time is None:
                    face_detected_after_pause_time = time.time()
                elif paused and face_detected_after_pause_time and time.time() - face_detected_after_pause_time >= required_continuous_detection_time_after_pause:
                    paused = False
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from the beginning
                    pygame.mixer.music.play()  # Play audio from the beginning
                    print("Resuming Video from Beginning")
                    face_detected_time = None
                    face_detected_after_pause_time = None

            # Reset no_face_start_time when a face is detected
            no_face_start_time = None

        else:
            face_detected_time = None
            face_detected_after_pause_time = None
            if no_face_start_time is None:
                no_face_start_time = time.time()
            elif time.time() - no_face_start_time >= 10:
                if not paused:
                    paused = True
                    pygame.mixer.music.pause()  # Pause audio
                    print("Pausing Video")

        # Draw face detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Display the webcam feed
        cv2.imshow('Webcam Feed', image)

        # Control video playback
        if not paused:
            ret, frame = video_cap.read()
            if not ret:
                print("End of Video")
                break
            cv2.imshow('Video Player', frame)
        else:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from the beginning
            cv2.imshow('Video Player', np.zeros((480, 640, 3), dtype=np.uint8))

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()
pygame.quit()