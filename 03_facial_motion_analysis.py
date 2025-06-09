import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def draw_flow(img, flow, points, step=16):
    """옵티컬 플로우 시각화"""
    h, w = flow.shape[:2]

    # 랜드마크 포인트에서의 플로우 계산
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    # 각 랜드마크 포인트에서 플로우 추출
    flow_vectors = []
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= y < h and 0 <= x < w:  # 이미지 범위 내 체크
            flow_x = fx[y, x]
            flow_y = fy[y, x]
            magnitude = np.sqrt(flow_x * flow_x + flow_y * flow_y)
            if magnitude > 0.5:  # 움직임이 있는 경우만
                flow_vectors.append((x, y, flow_x, flow_y, magnitude))

    # 움직임 강도의 최대값 계산
    if flow_vectors:
        max_magnitude = max(vector[4] for vector in flow_vectors)

        # 각 벡터 그리기
        for x, y, flow_x, flow_y, magnitude in flow_vectors:
            # 움직임 강도에 따른 색상 설정
            color_intensity = int((magnitude / max_magnitude) * 255)
            color = (0, color_intensity, 255)  # BGR

            # 화살표 끝점 계산
            end_x = int(x + flow_x * 2)  # 움직임을 더 잘 보이게 하기 위해 2배 스케일
            end_y = int(y + flow_y * 2)

            # 화살표 그리기
            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

            # 랜드마크 포인트 표시
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    return img


class FacialMotionAnalyzer:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Optical Flow 파라미터
        self.prev_gray = None
        self.prev_points = None

        # 안정화 파라미터
        self.initial_center = None
        self.STABLE_LANDMARKS = [168, 1, 2, 4, 5, 98, 327, 200, 151]  # 안정화에 사용할 랜드마크
        self.LEFT_EYE_LANDMARK = 33  # 왼쪽 눈 중심
        self.RIGHT_EYE_LANDMARK = 263  # 오른쪽 눈 중심

        # 움직임 기록을 위한 변수
        self.landmark_movements = defaultdict(list)  # 각 랜드마크별 움직임 기록
        self.frame_count = 0  # 처리된 프레임 수

    def stabilize_and_align_face(self, frame, landmarks):
        """얼굴 안정화 및 정렬"""
        h, w = frame.shape[:2]

        # 1. 얼굴 안정화 (Translation)
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.STABLE_LANDMARKS])
        current_center = np.mean(pts, axis=0)

        if self.initial_center is None:
            self.initial_center = current_center.copy()

        dx, dy = current_center - self.initial_center
        M_trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(frame, M_trans, (w, h))

        # 2. 얼굴 정렬 (Rotation)
        left_eye_pt = np.array([landmarks[self.LEFT_EYE_LANDMARK].x * w,
                                landmarks[self.LEFT_EYE_LANDMARK].y * h])
        right_eye_pt = np.array([landmarks[self.RIGHT_EYE_LANDMARK].x * w,
                                 landmarks[self.RIGHT_EYE_LANDMARK].y * h])

        # 안정화가 적용된 눈 좌표
        left_eye_stb = left_eye_pt - np.array([dx, dy])
        right_eye_stb = right_eye_pt - np.array([dx, dy])

        # 두 눈 중앙점 계산
        eye_center = (left_eye_stb + right_eye_stb) / 2.0
        cx, cy = float(eye_center[0]), float(eye_center[1])

        # 눈선 각도 계산
        dy_eyes = right_eye_stb[1] - left_eye_stb[1]
        dx_eyes = right_eye_stb[0] - left_eye_stb[0]
        angle = np.degrees(np.arctan2(dy_eyes, dx_eyes))

        # 회전 행렬 적용
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned = cv2.warpAffine(stabilized, M_rot, (w, h))

        # 변환 행렬 결합
        M_combined = np.vstack([M_rot, [0, 0, 1]]) @ np.vstack([M_trans, [0, 0, 1]])

        # 랜드마크 포인트 변환
        points = np.array([(lm.x * w, lm.y * h, 1) for lm in landmarks])
        transformed_points = (M_combined @ points.T).T
        transformed_points = transformed_points[:, :2]  # 호모그래피 좌표 제거

        return aligned, transformed_points

    def plot_top_movements(self, n=20):
        """상위 N개 랜드마크의 움직임을 막대 그래프로 표시"""
        # 평균 움직임 계산
        avg_movements = {}
        for landmark_idx, movements in self.landmark_movements.items():
            if movements:
                avg_movements[landmark_idx] = np.mean(movements)

        if not avg_movements:
            print("No movement data available")
            return

        # 움직임이 큰 순서대로 정렬하여 상위 N개 선택
        sorted_movements = sorted(
            avg_movements.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        # 그래프 생성
        plt.figure(figsize=(15, 6))
        landmarks = [str(idx) for idx, _ in sorted_movements]
        movements = [mov for _, mov in sorted_movements]

        plt.bar(landmarks, movements)
        plt.title(f'Top {n} Landmarks with Highest Movement')
        plt.xlabel('Landmark Index')
        plt.ylabel('Average Movement Magnitude')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 그래프 저장
        plt.savefig('landmark_movements_graph.png')
        plt.close()

    def process_frame(self, frame):
        """프레임 처리"""
        # 프레임 복사
        flow_frame = frame.copy()

        # 얼굴 검출
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 얼굴 안정화 및 정렬
            stabilized_frame, transformed_points = self.stabilize_and_align_face(frame, landmarks)

            # 현재 프레임을 그레이스케일로 변환
            current_gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is not None and self.prev_points is not None:
                try:
                    # 옵티컬 플로우 계산
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray,
                        current_gray,
                        None,
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )

                    # 각 랜드마크의 움직임 기록
                    for i, point in enumerate(transformed_points):
                        x, y = int(point[0]), int(point[1])
                        if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                            flow_x = flow[y, x, 0]
                            flow_y = flow[y, x, 1]
                            magnitude = np.sqrt(flow_x * flow_x + flow_y * flow_y)
                            self.landmark_movements[i].append(magnitude)

                    # 옵티컬 플로우 시각화
                    flow_frame = draw_flow(stabilized_frame.copy(), flow, transformed_points)

                except cv2.error as e:
                    print(f"Optical flow calculation error: {e}")

            # 현재 프레임 저장
            self.prev_gray = current_gray
            self.prev_points = transformed_points
            self.frame_count += 1

            return flow_frame

        return flow_frame

    def save_movements_to_csv(self, output_file='landmark_movements.csv'):
        """랜드마크별 평균 움직임을 CSV 파일로 저장하고 시각화"""
        # 각 랜드마크의 평균 움직임 계산
        avg_movements = {}
        for landmark_idx, movements in self.landmark_movements.items():
            if movements:  # 움직임이 기록된 경우만
                avg_movements[landmark_idx] = np.mean(movements)

        # DataFrame 생성 및 CSV 저장
        df = pd.DataFrame({
            'Landmark_Index': list(avg_movements.keys()),
            'Average_Movement': list(avg_movements.values())
        })
        df = df.sort_values('Average_Movement', ascending=False)  # 움직임이 큰 순서대로 정렬
        df.to_csv(output_file, index=False)
        print(f"Landmark movements saved to {output_file}")

        # 움직임 시각화
        self.plot_top_movements()


def main():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    analyzer = FacialMotionAnalyzer()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            flow = analyzer.process_frame(frame)

            # 결과 표시
            cv2.imshow('Facial Motion Analysis', flow)

            # ESC 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 종료 시 CSV 파일 저장 및 시각화
        analyzer.save_movements_to_csv()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()