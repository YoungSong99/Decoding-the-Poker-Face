import cv2
import mediapipe as mp

# ------------------------------------------------------------
# (A) MediaPipe FaceMesh & Drawing 유틸 초기화
# ------------------------------------------------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,       # 눈꺼풀, 입술 등 세분화된 478개 포인트
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# (B) 웹캠 열기
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# ------------------------------------------------------------
# (C) 메인 루프: 프레임마다 FaceMesh → 랜드마크 드로잉 → 표시
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (C-1) BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # (C-2) FaceMesh 추론
    results = face_mesh.process(rgb)

    # (C-3) 랜드마크 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 전체 랜드마크 점만 그리려면:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=None,  # 점만 표시
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,255,0), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=None
            )
            # ── 만약 연결선(468개 지점 간 사슬)도 같이 그리고 싶다면:
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face.FACE_CONNECTIONS,
            #     landmark_drawing_spec=mp_style.get_default_face_mesh_landmarks_style(),
            #     connection_drawing_spec=mp_style.get_default_face_mesh_contours_style()
            # )

    # (C-4) 화면 출력
    cv2.imshow("Webcam FaceMesh Landmarks", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------------------------------------------------
# (D) 정리
# ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
