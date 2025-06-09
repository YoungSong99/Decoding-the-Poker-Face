import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Blink counter variables
blink_count = 0
ear_threshold = 0.22  # Eye aspect ratio threshold for blink detection
consecutive_frames = 3  # Number of consecutive frames to confirm a blink
frame_count = 0

eye_closed = False  # To track if the eyes are currently closed

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face mesh landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Extract eye landmarks
            left_eye_landmarks = np.array([
                (face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h)
                for i in [33, 160, 158, 133, 153, 144]
            ])
            right_eye_landmarks = np.array([
                (face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h)
                for i in [263, 387, 385, 362, 380, 373]
            ])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0

            # Blink detection
            if ear < ear_threshold:
                frame_count += 1
                if frame_count >= consecutive_frames and not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                frame_count = 0
                eye_closed = False

            # Define the text properties
            text = f'Blink Count: {blink_count}'
            ear_text = f'EAR: {ear:.2f}'

            # Set font and size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2  # Increased font scale for larger text
            font_thickness = 3  # Increased thickness for better visibility

            # Calculate text size for centering
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            (ear_text_width, ear_text_height), _ = cv2.getTextSize(ear_text, font, font_scale, font_thickness)

            # Positioning text
            text_x = 30  # Adjust as needed for your screen
            text_y = 50
            ear_text_x = 30
            ear_text_y = 100

            # Draw background rectangles for the text
            cv2.rectangle(frame, (text_x, text_y - text_height - 10),
                          (text_x + text_width + 20, text_y + 10),
                          (147, 20, 255), -1)  # Pink background
            cv2.rectangle(frame, (ear_text_x, ear_text_y - ear_text_height - 10),
                          (ear_text_x + ear_text_width + 20, ear_text_y + 10),
                          (147, 20, 255), -1)  # Pink background

            # Put the text on the frame with white color and bold
            cv2.putText(frame, text, (text_x + 10, text_y), font, font_scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(frame, ear_text, (ear_text_x + 10, ear_text_y), font, font_scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)

    else:
        cv2.putText(frame, 'No face detected', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print('No face detected')

    # Display the frame
    cv2.imshow('Eye Blink Counter', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()