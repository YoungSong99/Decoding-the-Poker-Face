"""
Webcam Heart Rate Monitor (Forehead ROI with MediaPipe, refined to exclude eyes)
Gilad Oved (modified to use multiple MediaPipe FaceMesh landmarks for precise forehead detection)
December 2018 (modified 2025)
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
from datetime import datetime
import matplotlib.pyplot as plt

class HRVAnalyzer:
    def __init__(self, window_size=45):  # 3초 영상 기준으로 수정 (15fps * 3s = 45 frames)
        self.window_size = window_size
        self.timestamps = []
        self.bpm_values = []
        self.rr_intervals = []
        self.time_domain_results = []
        self.freq_domain_results = []
        
    def add_bpm(self, bpm, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        self.timestamps.append(timestamp)
        self.bpm_values.append(bpm)
        
        # BPM을 RR 간격으로 변환 (밀리초 단위)
        rr_interval = (60.0 / bpm) * 1000
        self.rr_intervals.append(rr_interval)
        
        # 윈도우 크기 유지
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
            self.bpm_values.pop(0)
            self.rr_intervals.pop(0)
    
    def calculate_time_domain_hrv(self):
        """시간 도메인 HRV 지표 계산"""
        if len(self.rr_intervals) < 2:
            return None
            
        results = {
            'SDNN': np.std(self.rr_intervals),  # RR 간격의 표준편차
            'RMSSD': np.sqrt(np.mean(np.diff(self.rr_intervals) ** 2)),  # 연속된 RR 간격 차이의 제곱평균의 제곱근
            'NN50': sum(abs(np.diff(self.rr_intervals)) > 50),  # 50ms 이상 차이나는 연속 RR 간격의 수
            'pNN50': (sum(abs(np.diff(self.rr_intervals)) > 50) / len(self.rr_intervals)) * 100  # NN50의 비율
        }
        return results
    
    def calculate_frequency_domain_hrv(self):
        """주파수 도메인 HRV 지표 계산"""
        if len(self.rr_intervals) < 10:  # 최소 데이터 요구사항
            return None
            
        # RR 간격을 리샘플링 (4Hz)
        rr_x = np.cumsum(self.rr_intervals) / 1000.0  # 초 단위로 변환
        rr_y = self.rr_intervals
        fs = 4.0  # 리샘플링 주파수
        rr_x_new = np.arange(rr_x[0], rr_x[-1], 1/fs)
        rr_y_new = np.interp(rr_x_new, rr_x, rr_y)
        
        # 선형 트렌드 제거
        rr_y_detrend = signal.detrend(rr_y_new)
        
        # 주파수 영역 파워 계산
        frequencies, powers = signal.welch(rr_y_detrend, fs=fs, nperseg=len(rr_y_detrend))
        
        # 주파수 대역별 파워 계산
        vlf_mask = (frequencies >= 0.0033) & (frequencies < 0.04)  # Very Low Frequency
        lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)    # Low Frequency
        hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)     # High Frequency
        
        vlf_power = np.trapz(powers[vlf_mask], frequencies[vlf_mask])
        lf_power = np.trapz(powers[lf_mask], frequencies[lf_mask])
        hf_power = np.trapz(powers[hf_mask], frequencies[hf_mask])
        total_power = vlf_power + lf_power + hf_power
        
        results = {
            'VLF': vlf_power,
            'LF': lf_power,
            'HF': hf_power,
            'LF/HF': lf_power / hf_power if hf_power > 0 else 0,
            'Total Power': total_power
        }
        return results
    
    def plot_hrv(self, output_path='hrv_analysis.png'):
        """HRV 분석 결과 시각화"""
        if len(self.rr_intervals) < 2:
            return
            
        plt.figure(figsize=(15, 10))
        
        # 1. RR 간격 시계열
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, self.rr_intervals)
        plt.title('RR Intervals Over Time')
        plt.ylabel('RR Interval (ms)')
        
        # 2. RR 간격 분포
        plt.subplot(3, 1, 2)
        plt.hist(self.rr_intervals, bins=30, density=True)
        plt.title('RR Interval Distribution')
        plt.xlabel('RR Interval (ms)')
        
        # 3. 파워 스펙트럼
        if len(self.rr_intervals) >= 10:
            plt.subplot(3, 1, 3)
            frequencies, powers = signal.welch(signal.detrend(self.rr_intervals), fs=4.0)
            plt.semilogy(frequencies, powers)
            plt.title('Power Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            
            # 주파수 대역 표시
            plt.axvline(0.04, color='r', linestyle='--', alpha=0.5)
            plt.axvline(0.15, color='r', linestyle='--', alpha=0.5)
            plt.text(0.02, plt.ylim()[1], 'VLF')
            plt.text(0.095, plt.ylim()[1], 'LF')
            plt.text(0.275, plt.ylim()[1], 'HF')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_final_analysis(self, output_path='final_analysis.png'):
        """카메라 종료 후 BPM과 HRV 분석 결과 시각화"""
        if len(self.rr_intervals) < 2:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 1. BPM 변화
        plt.subplot(2, 1, 1)
        times = [(t - self.timestamps[0]).total_seconds() for t in self.timestamps]
        plt.plot(times, self.bpm_values, 'b-', linewidth=2, label='Heart Rate')
        plt.title('Heart Rate Changes')
        plt.xlabel('Time (seconds)')
        plt.ylabel('BPM')
        plt.grid(True)
        
        # 2. SDNN (HRV) 변화
        if self.time_domain_results:
            plt.subplot(2, 1, 2)
            sdnn_values = [result['SDNN'] for result in self.time_domain_results]
            analysis_times = range(len(self.time_domain_results))
            
            plt.plot(analysis_times, sdnn_values, 'r-', linewidth=2, label='SDNN')
            plt.title('Heart Rate Variability (SDNN)')
            plt.xlabel('Analysis Window')
            plt.ylabel('SDNN (ms)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 간단한 통계 저장
        self._save_simple_statistics()

    def _save_simple_statistics(self):
        """간단한 통계 요약을 텍스트 파일로 저장"""
        with open('heart_statistics.txt', 'w', encoding='utf-8') as f:
            f.write("=== 심박수 통계 ===\n")
            f.write(f"평균 BPM: {np.mean(self.bpm_values):.1f}\n")
            f.write(f"BPM 범위: {np.min(self.bpm_values):.1f} - {np.max(self.bpm_values):.1f}\n\n")
            
            if self.time_domain_results:
                f.write("=== HRV 통계 ===\n")
                sdnn_values = [result['SDNN'] for result in self.time_domain_results]
                f.write(f"평균 SDNN: {np.mean(sdnn_values):.1f} ms\n")
                f.write(f"SDNN 범위: {np.min(sdnn_values):.1f} - {np.max(sdnn_values):.1f} ms\n")
        
        print("\n✅ 분석 결과가 저장되었습니다:")
        print("- 그래프: 'final_analysis.png'")
        print("- 통계: 'heart_statistics.txt'")

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
    # Crop back to original detectionFrame size
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# ──────────────────────────────────────────────────────────────────────────────
# Mediapipe FaceMesh Initialization (for forehead detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
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
bufferSize = 45  # 3초 영상 기준 (15fps * 3s = 45 frames)
bufferIndex = 0

# Pre-allocate arrays for Gaussian pyramid and FFT buffering
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 5  # 더 자주 계산
bpmBufferIndex = 0
bpmBufferSize = 5  # 버퍼 크기 축소
bpmBuffer = np.zeros((bpmBufferSize))
hrv_analyzer = HRVAnalyzer(window_size=45)  # 3초 기준

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

# Landmark index sets for defining forehead region
# These indices come from MediaPipe FaceMesh topology.
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
    # Mediapipe: Detect face landmarks and extract a precise forehead ROI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # 1) Compute eyebrow line (to exclude eyes): find the minimal y among eyebrow landmarks
        eyebrow_ys = []
        for idx in eyebrow_indices:
            lm = landmarks[idx]
            eyebrow_ys.append(int(lm.y * h))
        y_eyebrow_min = min(eyebrow_ys)

        # 2) Compute temple x-coordinates (to set left/right bounds of forehead)
        temple_xs = []
        for idx in temple_indices:
            lm = landmarks[idx]
            temple_xs.append(int(lm.x * w))
        x_left = min(temple_xs)
        x_right = max(temple_xs)

        # 3) Define forehead patch height (from some distance above eyebrow line down to eyebrow line)
        forehead_height = int(0.25 * h)  # adjust this ratio as needed
        y2 = max(y_eyebrow_min - int(0.02 * h), 0)     # a small margin above eyebrows
        y1 = max(y2 - forehead_height, 0)             # top of forehead region

        # 4) Add horizontal margins around temples to capture entire forehead width
        margin_x = int(0.05 * w)
        x1 = max(x_left - margin_x, 0)
        x2 = min(x_right + margin_x, w)

        # 5) Crop the forehead region and resize to (videoWidth, videoHeight)
        forehead_roi = frame[y1:y2, x1:x2]
        if forehead_roi.size == 0:
            # fallback to central crop if something went wrong
            detectionFrame = frame[
                videoHeight // 2 : realHeight - videoHeight // 2,
                videoWidth  // 2 : realWidth  - videoWidth  // 2,
                :
            ]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
        else:
            detectionFrame = cv2.resize(forehead_roi, (videoWidth, videoHeight))
    else:
        # Fallback: center-crop if no face detected
        detectionFrame = frame[
            videoHeight // 2 : realHeight - videoHeight // 2,
            videoWidth  // 2 : realWidth  - videoWidth  // 2,
            :
        ]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
    # ──────────────────────────────────────────────────────────────────────────

    # Construct Gaussian Pyramid on detectionFrame
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    # Bandpass Filter: zero out frequencies outside the desired heart-rate band
    fourierTransform[mask == False] = 0

    # Grab a Pulse: compute average magnitude in each frequency bin periodically
    if bufferIndex % bpmCalculationFrequency == 0:
        i += 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify: inverse FFT and scale by alpha
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    # Reconstruct Resulting Frame from the pyramid
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    # Overlay the amplified forehead output onto the original frame
    frame[
        videoHeight // 2 : realHeight - videoHeight // 2,
        videoWidth  // 2 : realWidth  - videoWidth  // 2,
        :
    ] = outputFrame

    # Draw bounding box corresponding to the overlay region
    cv2.rectangle(
        frame,
        (videoWidth // 2, videoHeight // 2),
        (realWidth - videoWidth // 2, realHeight - videoHeight // 2),
        boxColor, boxWeight
    )

    # Display BPM text and calculate HRV once enough frames have been processed
    if i > bpmBufferSize:
        avg_bpm = int(bpmBuffer.mean())
        hrv_analyzer.add_bpm(avg_bpm)
        
        # 매 프레임마다 HRV 분석 수행 (짧은 영상이므로)
        time_domain_hrv = hrv_analyzer.calculate_time_domain_hrv()
        freq_domain_hrv = hrv_analyzer.calculate_frequency_domain_hrv()
        
        if time_domain_hrv:
            # 분석 결과 저장
            hrv_analyzer.time_domain_results.append(time_domain_hrv)
            
            # HRV 지표 화면에 표시
            cv2.putText(
                frame,
                f"SDNN: {time_domain_hrv['SDNN']:.1f}ms",
                (20, 60),
                font,
                fontScale,
                fontColor,
                lineType
            )
        
        cv2.putText(
            frame,
            "BPM: %d" % avg_bpm,
            bpmTextLocation,
            font,
            fontScale,
            fontColor,
            lineType
        )
    else:
        cv2.putText(
            frame,
            "Calculating BPM...",
            loadingTextLocation,
            font,
            fontScale,
            fontColor,
            lineType
        )

    outputVideoWriter.write(frame)

    # Show live preview window if not writing to file only
    if len(sys.argv) != 2:
        cv2.imshow("Webcam Heart Rate Monitor (Forehead)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()

# 전체 데이터 분석 결과 시각화
hrv_analyzer.plot_final_analysis()
