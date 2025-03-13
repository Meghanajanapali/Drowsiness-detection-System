import cv2
import dlib
import imutils
import numpy as np
import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer

# Initialize sound alert
mixer.init()
mixer.music.load("music.wav")

# Compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Constants
THRESH = 0.25  # EAR threshold for closed eyes
TIME_LIMIT = 2  # Time in seconds before alert is triggered

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get indexes for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)

# Read the first frame and resize it
ret, fixed_frame = cap.read()
if ret:
    fixed_frame = imutils.resize(fixed_frame, width=450)
    fixed_height, fixed_width = fixed_frame.shape[:2]  # Store fixed dimensions

start_time = None  # Time when eyes were first detected as closed
alert_playing = False  # Prevent repeated alerts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the current frame to match the fixed frame size
    frame = imutils.resize(frame, width=fixed_width, height=fixed_height)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    subjects = detector(gray, 0)

    if subjects:
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # Extract eye coordinates
            leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]

            # Compute EAR for both eyes
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            # Check if eyes are closed
            if ear < THRESH:
                if start_time is None:
                    start_time = time.time()  # Start counting when eyes first close

                elapsed_time = time.time() - start_time  # Calculate time eyes remain closed

                if elapsed_time >= TIME_LIMIT and not alert_playing:
                    # Draw a red alert box and text once
                    cv2.rectangle(frame, (50, 20), (400, 60), (0, 0, 255), -1)
                    cv2.putText(frame, "!!! ALERT !!!", (120, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mixer.music.play()
                    alert_playing = True
            else:
                start_time = None  # Reset timer when eyes open
                alert_playing = False  # Allow alert to play again if needed

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
