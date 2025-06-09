import cv2
from face_alignment import FaceAligner  # FaceAligner가 정의된 파일에서 import

def main():
    cap = cv2.VideoCapture(0)
    aligner = FaceAligner()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 정렬
        aligned, landmarks = aligner.align(frame)

        # 정렬된 얼굴이 있으면 화면에 표시
        if aligned is not None:
            cv2.imshow('Aligned Face', aligned)
        else:
            cv2.imshow('Aligned Face', frame)

        # 원본 프레임도 같이 보기 원하면 아래 코드 추가
        # cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()