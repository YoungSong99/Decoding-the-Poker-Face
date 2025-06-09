"""
Webcam Heart Rate Monitor with Face Stabilization
(Combines face alignment, stabilization, and heart rate monitoring)
"""

import sys
import cv2
import numpy as np
import mediapipe as mp

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# ──────────────────────────────────────────────────────────────────────────────
# Mediapipe FaceMesh Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face Stabilization Parameters
STABLE_LANDMARKS = [168, 1, 2, 4, 5, 98, 327, 200, 151]  # 안정화용 랜드마크
LEFT_EYE_LANDMARK = 33    # 왼쪽 눈 중심
RIGHT_EYE_LANDMARK = 263  # 오른쪽 눈 중심
initial_center = None     # 첫 프레임 기준점
# ──────────────────────────────────────────────────────────────────────────────

# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)

realWidth = 320
realHeight = 240
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)

# Output Videos
if len(sys.argv) != 2:
    originalVideoFilename = "original.mov"
    originalVideoWriter = cv2.VideoWriter()
    originalVideoWriter.open(
        originalVideoFilename,
        cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'),
        videoFrameRate,
        (realWidth, realHeight),
        True
    )

outputVideoFilename = "output.mov"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(
    outputVideoFilename,
    cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'),
    videoFrameRate,
    (realWidth, realHeight),
    True
)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Pre-allocate arrays
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 30
bpmBufferIndex = 0
bpmBufferSize = 20
bpmBuffer = np.zeros((bpmBufferSize))

i = 0  # Frame counter for BPM averaging

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth // 2 + 5, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Landmark index sets for forehead region
eyebrow_indices = [
    63, 105, 52, 65, 55,   # left eyebrow
    295, 334, 282, 283, 276 # right eyebrow
]
temple_indices = [127, 356]  # approximate left & right temple/forehead edges

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # If writing original video (no file argument), copy and write the raw frame
    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    # ──────────────────────────────────────────────────────────────────────────
    # Face Detection and Stabilization
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # 1) Face Stabilization
        pts = np.array([[lm.x * w, lm.y * h] for lm in [landmarks[i] for i in STABLE_LANDMARKS]])
        current_center = np.mean(pts, axis=0)

        if initial_center is None:
            initial_center = current_center.copy()

        dx, dy = current_center - initial_center
        M_trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(frame, M_trans, (w, h))

        # 2) Face Alignment
        left_eye_pt = np.array([landmarks[LEFT_EYE_LANDMARK].x * w,
                               landmarks[LEFT_EYE_LANDMARK].y * h])
        right_eye_pt = np.array([landmarks[RIGHT_EYE_LANDMARK].x * w,
                                landmarks[RIGHT_EYE_LANDMARK].y * h])

        left_eye_stb = left_eye_pt - np.array([dx, dy])
        right_eye_stb = right_eye_pt - np.array([dx, dy])

        eye_center = (left_eye_stb + right_eye_stb) / 2.0
        cx, cy = float(eye_center[0]), float(eye_center[1])

        dy_eyes = right_eye_stb[1] - left_eye_stb[1]
        dx_eyes = right_eye_stb[0] - left_eye_stb[0]
        angle = np.degrees(np.arctan2(dy_eyes, dx_eyes))

        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned_frame = cv2.warpAffine(stabilized, M_rot, (w, h))

        # 3) Extract Forehead ROI from aligned frame
        # Compute eyebrow line
        eyebrow_ys = []
        for idx in eyebrow_indices:
            lm = landmarks[idx]
            eyebrow_ys.append(int(lm.y * h - dy))  # Adjust for stabilization
        y_eyebrow_min = min(eyebrow_ys)

        # Compute temple x-coordinates
        temple_xs = []
        for idx in temple_indices:
            lm = landmarks[idx]
            temple_xs.append(int(lm.x * w - dx))  # Adjust for stabilization
        x_left = min(temple_xs)
        x_right = max(temple_xs)

        # Define forehead patch
        forehead_height = int(0.1 * h)
        y2 = max(y_eyebrow_min - int(0.02 * h), 0)
        y1 = max(y2 - forehead_height, 0)
        margin_x = int(0.05 * w)
        x1 = max(x_left - margin_x, 0)
        x2 = min(x_right + margin_x, w)

        # Extract and resize ROI from aligned frame
        forehead_roi = aligned_frame[y1:y2, x1:x2]
        if forehead_roi.size > 0:
            detectionFrame = cv2.resize(forehead_roi, (videoWidth, videoHeight))
        else:
            detectionFrame = cv2.resize(aligned_frame[
                videoHeight // 2 : realHeight - videoHeight // 2,
                videoWidth // 2 : realWidth - videoWidth // 2
            ], (videoWidth, videoHeight))
    else:
        # Fallback: use original frame if no face detected
        detectionFrame = cv2.resize(frame[
            videoHeight // 2 : realHeight - videoHeight // 2,
            videoWidth // 2 : realWidth - videoWidth // 2
        ], (videoWidth, videoHeight))
        aligned_frame = frame.copy()
    # ──────────────────────────────────────────────────────────────────────────

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    # Bandpass Filter
    fourierTransform[mask == False] = 0

    # Calculate Heart Rate
    if bufferIndex % bpmCalculationFrequency == 0:
        i += 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify and reconstruct
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    # Overlay the amplified forehead output onto the aligned frame
    aligned_frame[
        videoHeight // 2 : realHeight - videoHeight // 2,
        videoWidth // 2 : realWidth - videoWidth // 2
    ] = outputFrame

    # Draw bounding box
    cv2.rectangle(
        aligned_frame,
        (videoWidth // 2, videoHeight // 2),
        (realWidth - videoWidth // 2, realHeight - videoHeight // 2),
        boxColor, boxWeight
    )

    # Display BPM
    if i > bpmBufferSize:
        avg_bpm = int(bpmBuffer.mean())
        cv2.putText(
            aligned_frame,
            "BPM: %d" % avg_bpm,
            bpmTextLocation,
            font,
            fontScale,
            fontColor,
            lineType
        )
    else:
        cv2.putText(
            aligned_frame,
            "Calculating BPM...",
            loadingTextLocation,
            font,
            fontScale,
            fontColor,
            lineType
        )

    outputVideoWriter.write(aligned_frame)

    # Show preview
    if len(sys.argv) != 2:
        cv2.imshow("Heart Rate Monitor (Stabilized)", aligned_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()
