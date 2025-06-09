import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy import stats
import time

# 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 랜드마크 인덱스 정의
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_LEFT_CORNER = 263
RIGHT_EYE_RIGHT_CORNER = 33
LEFT_MOUTH_CORNER = 287
RIGHT_MOUTH_CORNER = 57
LEFT_PUPIL = 468
RIGHT_PUPIL = 473

# 시선 데이터 저장
gaze_trail = deque(maxlen=100)  # 실시간 시각화용
gaze_data = []  # 전체 데이터 저장용

class GazePoint:
    def __init__(self, x, y, timestamp):
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.type = None  # 'fixation' 또는 'saccade'
        self.duration = 0  # fixation 지속 시간

def relative(landmark, shape):
    """Convert normalized landmark to image coordinates"""
    return (
        int(landmark.x * shape[1]),
        int(landmark.y * shape[0])
    )

def relativeT(landmark, shape):
    """Convert normalized landmark to image coordinates with Z"""
    return (
        int(landmark.x * shape[1]),
        int(landmark.y * shape[0]),
        0
    )

def calculate_gaze(frame, face_landmarks):
    """Calculate and visualize gaze direction"""
    h, w = frame.shape[:2]
    
    # 2D image points
    image_points = np.array([
        relative(face_landmarks.landmark[NOSE_TIP], frame.shape),
        relative(face_landmarks.landmark[CHIN], frame.shape),
        relative(face_landmarks.landmark[LEFT_EYE_LEFT_CORNER], frame.shape),
        relative(face_landmarks.landmark[RIGHT_EYE_RIGHT_CORNER], frame.shape),
        relative(face_landmarks.landmark[LEFT_MOUTH_CORNER], frame.shape),
        relative(face_landmarks.landmark[RIGHT_MOUTH_CORNER], frame.shape)
    ], dtype="double")
    
    # 2D image points with Z=0
    image_points1 = np.array([
        relativeT(face_landmarks.landmark[NOSE_TIP], frame.shape),
        relativeT(face_landmarks.landmark[CHIN], frame.shape),
        relativeT(face_landmarks.landmark[LEFT_EYE_LEFT_CORNER], frame.shape),
        relativeT(face_landmarks.landmark[RIGHT_EYE_RIGHT_CORNER], frame.shape),
        relativeT(face_landmarks.landmark[LEFT_MOUTH_CORNER], frame.shape),
        relativeT(face_landmarks.landmark[RIGHT_MOUTH_CORNER], frame.shape)
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye left corner
        (43.3, 32.7, -26),  # Right eye right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    # Camera matrix estimation
    focal_length = frame.shape[1]
    center = (frame.shape[1]/2, frame.shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Solve for pose
    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Get pupil coordinates and calculate gaze
    left_pupil = relative(face_landmarks.landmark[LEFT_PUPIL], frame.shape)
    right_pupil = relative(face_landmarks.landmark[RIGHT_PUPIL], frame.shape)

    # Calculate 3D gaze
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)

    if transformation is not None:
        # Project left pupil to 3D
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        
        # Calculate 3D gaze point
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10
        
        # Project 3D gaze direction onto image plane
        (eye_pupil2D, _) = cv2.projectPoints(
            (int(S[0]), int(S[1]), int(S[2])),
            rotation_vector, translation_vector,
            camera_matrix, dist_coeffs
        )
        
        # Project head pose
        (head_pose, _) = cv2.projectPoints(
            (int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
            rotation_vector, translation_vector,
            camera_matrix, dist_coeffs
        )
        
        # Correct gaze for head rotation
        gaze = np.array(left_pupil) + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)
        
        # Draw gaze line
        p1 = (int(left_pupil[0]), int(left_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.circle(frame, p1, 3, (255, 0, 0), -1)
        
        return gaze
    return None

def draw_gaze_trail(frame, trail):
    """Draw gaze trail"""
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        p2 = (int(trail[i][0]), int(trail[i][1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
    return frame

def analyze_gaze_pattern(gaze_data, velocity_threshold=50, duration_threshold=100):
    """Analyze gaze data to identify fixations and saccades"""
    if len(gaze_data) < 2:
        return
    
    # 속도 계산
    for i in range(1, len(gaze_data)):
        point1 = gaze_data[i-1]
        point2 = gaze_data[i]
        distance = euclidean([point1.x, point1.y], [point2.x, point2.y])
        time_diff = (point2.timestamp - point1.timestamp) * 1000  # ms 단위로 변환
        velocity = distance / time_diff if time_diff > 0 else 0
        
        # Fixation vs Saccade 분류
        if velocity < velocity_threshold:
            point2.type = 'fixation'
            # Fixation 지속 시간 계산
            if point1.type == 'fixation':
                point2.duration = point1.duration + time_diff
            else:
                point2.duration = time_diff
        else:
            point2.type = 'saccade'
            point2.duration = 0

def plot_fixation_saccade(gaze_data, frame_size):
    """Create gaze direction plot in X-Y coordinate space with arrows"""
    if len(gaze_data) < 2:
        return
    
    # 데이터 준비
    x_coords = [p.x for p in gaze_data]
    y_coords = [p.y for p in gaze_data]
    
    # 플롯 생성
    plt.figure(figsize=(12, 8))
    
    # 시선 이동 경로를 화살표로 표시
    for i in range(1, len(gaze_data)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        
        # 너무 짧은 움직임은 건너뛰기 (노이즈 제거)
        if abs(dx) < 5 and abs(dy) < 5:
            continue
            
        plt.arrow(x_coords[i-1], y_coords[i-1], dx, dy,
                 head_width=20, head_length=20,
                 fc='blue', ec='blue', alpha=0.6,
                 length_includes_head=True)
    
    # 시작점과 끝점 표시
    plt.plot(x_coords[0], y_coords[0], 'go', label='Start', markersize=10)
    plt.plot(x_coords[-1], y_coords[-1], 'ro', label='End', markersize=10)
    
    # 축 설정
    plt.xlim(0, frame_size[0])
    plt.ylim(frame_size[1], 0)  # y축 반전 (화면 좌표계와 일치)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Gaze Direction Trail')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 화면 비율 유지
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 저장
    plt.savefig('gaze_direction_xy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ 시선 방향 플롯을 'gaze_direction_xy.png'로 저장했습니다.")

def main():
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_output = None
    start_time = time.time()
    
    print("Press 'q' to quit and generate the Fixation-Saccade plot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            gaze_point = calculate_gaze(frame, results.multi_face_landmarks[0])
            
            if gaze_point is not None:
                # 실시간 시각화용
                gaze_trail.append(gaze_point)
                frame = draw_gaze_trail(frame, gaze_trail)
                last_output = frame.copy()
                
                # 전체 데이터 저장
                current_time = time.time()
                gaze_data.append(GazePoint(
                    x=gaze_point[0],
                    y=gaze_point[1],
                    timestamp=current_time
                ))
        
        cv2.imshow("3D Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 시선 패턴 분석 및 시각화
    if len(gaze_data) > 0:
        analyze_gaze_pattern(gaze_data)
        plot_fixation_saccade(gaze_data, (w, h))
        
        if last_output is not None:
            cv2.imwrite("gaze_trail.png", last_output)
            print("✅ 시선 궤적을 'gaze_trail.png'로 저장했습니다.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
