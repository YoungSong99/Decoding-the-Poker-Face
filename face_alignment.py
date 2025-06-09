import cv2
import mediapipe as mp
import numpy as np

class FaceAligner:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.STABLE_LANDMARKS = [168, 1, 2, 4, 5, 98, 327, 200, 151]
        self.LEFT_EYE_LANDMARK = 33
        self.RIGHT_EYE_LANDMARK = 263
        self.initial_center = None

    def align(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame, None
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in self.STABLE_LANDMARKS])
        current_center = np.mean(pts, axis=0)
        if self.initial_center is None:
            self.initial_center = current_center.copy()
        dx, dy = current_center - self.initial_center
        M_trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(frame, M_trans, (w, h))
        left_eye_pt = np.array([lm[self.LEFT_EYE_LANDMARK].x * w, lm[self.LEFT_EYE_LANDMARK].y * h])
        right_eye_pt = np.array([lm[self.RIGHT_EYE_LANDMARK].x * w, lm[self.RIGHT_EYE_LANDMARK].y * h])
        left_eye_stb = left_eye_pt - np.array([dx, dy])
        right_eye_stb = right_eye_pt - np.array([dx, dy])
        eye_center = (left_eye_stb + right_eye_stb) / 2.0
        cx, cy = float(eye_center[0]), float(eye_center[1])
        dy_eyes = right_eye_stb[1] - left_eye_stb[1]
        dx_eyes = right_eye_stb[0] - left_eye_stb[0]
        angle = np.degrees(np.arctan2(dy_eyes, dx_eyes))
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned = cv2.warpAffine(stabilized, M_rot, (w, h))
        return aligned, lm 