import librosa
import librosa.display
import numpy as np
import parselmouth
from parselmouth.praat import call
import scipy.stats as stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import os
import webrtcvad
import struct
import wave
from array import array
from scipy.io import wavfile

class VoiceAnalyzer:
    def __init__(self, input_path, sr=None):
        """
        음성 분석을 위한 클래스 초기화
        :param input_path: 비디오 또는 오디오 파일 경로
        :param sr: 샘플링 레이트 (None일 경우 원본 유지)
        """
        self.input_path = input_path
        self.audio_path = self._extract_audio() if self._is_video_file() else input_path
        
        # librosa로 오디오 로드
        self.y, self.sr = librosa.load(self.audio_path, sr=sr if sr else 16000)  # VAD를 위해 16kHz로 설정
        
        # 음성 구간 감지 및 노이즈 제거
        self.y = self._isolate_voice()
        
        # Praat 사운드 객체 생성 (음성 구간만)
        temp_voice_path = 'temp_voice.wav'
        wavfile.write(temp_voice_path, self.sr, self.y)
        self.sound = parselmouth.Sound(temp_voice_path)
        os.remove(temp_voice_path)
        
        # 임시 오디오 파일 삭제 (비디오에서 추출한 경우)
        if self._is_video_file():
            os.remove(self.audio_path)
    
    def _is_video_file(self):
        """파일이 비디오인지 확인"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        return any(self.input_path.lower().endswith(ext) for ext in video_extensions)
    
    def _extract_audio(self):
        """비디오에서 오디오 추출"""
        print("비디오에서 오디오 추출 중...")
        
        # 임시 오디오 파일 경로 생성
        base_name = os.path.splitext(self.input_path)[0]
        temp_audio_path = f"{base_name}_temp_audio.wav"
        
        try:
            # 비디오에서 오디오 추출
            video = VideoFileClip(self.input_path)
            audio = video.audio
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            video.close()
            print("✅ 오디오 추출 완료")
            
            return temp_audio_path
            
        except Exception as e:
            print(f"❌ 오디오 추출 실패: {str(e)}")
            raise
    
    def _isolate_voice(self, aggressiveness=3, frame_duration=30):
        """
        VAD를 사용하여 음성 구간만 분리
        :param aggressiveness: VAD 감도 (0-3, 높을수록 엄격)
        :param frame_duration: 프레임 길이 (ms)
        """
        print("음성 구간 분리 중...")
        
        # VAD 초기화
        vad = webrtcvad.Vad(aggressiveness)
        
        # 오디오를 프레임으로 분할
        samples_per_frame = int(self.sr * frame_duration / 1000)
        frames = []
        voice_frames = []
        
        # 16비트 PCM으로 변환
        audio_samples = np.int16(self.y * 32768)
        
        # 프레임 단위로 처리
        for start in range(0, len(audio_samples), samples_per_frame):
            frame = audio_samples[start:start + samples_per_frame]
            if len(frame) < samples_per_frame:
                break
            
            # VAD 적용
            frame_bytes = struct.pack("%dh" % len(frame), *frame)
            frames.append(frame)
            
            # 음성 구간 감지
            try:
                is_speech = vad.is_speech(frame_bytes, self.sr)
                voice_frames.append(is_speech)
            except:
                voice_frames.append(False)
        
        # 음성 구간 부드럽게 처리 (중간값 필터)
        window_size = 5
        smoothed_voice_frames = []
        for i in range(len(voice_frames)):
            start = max(0, i - window_size // 2)
            end = min(len(voice_frames), i + window_size // 2 + 1)
            smoothed_voice_frames.append(sum(voice_frames[start:end]) > (end - start) // 2)
        
        # 음성 구간만 추출
        voice_signal = np.zeros_like(self.y[:len(frames) * samples_per_frame])
        for i, is_voice in enumerate(smoothed_voice_frames):
            if is_voice:
                start = i * samples_per_frame
                end = start + samples_per_frame
                voice_signal[start:end] = self.y[start:end]
        
        print("✅ 음성 구간 분리 완료")
        return voice_signal
    
    def analyze_voice_tension(self):
        """
        음성 긴장도 분석
        - 스펙트럴 센트로이드: 음색의 "밝기"를 나타냄
        - 스펙트럴 롤오프: 주파수 분포의 형태를 나타냄
        """
        # 스펙트럴 센트로이드 계산
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        cent_mean = np.mean(cent)
        
        # 스펙트럴 롤오프 계산
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        rolloff_mean = np.mean(rolloff)
        
        # 음성 에너지 계산
        rms = librosa.feature.rms(y=self.y)
        rms_mean = np.mean(rms)
        
        tension_score = {
            'spectral_centroid': float(cent_mean),
            'spectral_rolloff': float(rolloff_mean),
            'energy': float(rms_mean)
        }
        
        return tension_score
    
    def analyze_speech_rate_and_pauses(self, min_silence_duration=0.3):
        """
        말하기 속도와 휴지(pause) 분석
        :param min_silence_duration: 최소 무음 구간 길이 (초)
        """
        # VAD로 이미 음성 구간이 분리되어 있으므로, 
        # 0에 가까운 값을 가진 구간을 휴지로 간주
        silence_threshold = np.max(np.abs(self.y)) * 0.01  # 최대 진폭의 1%
        
        # 무음 구간 찾기
        is_silence = np.abs(self.y) < silence_threshold
        
        # 연속된 무음 구간 찾기
        silence_regions = []
        start = None
        for i, silent in enumerate(is_silence):
            if silent and start is None:
                start = i
            elif not silent and start is not None:
                duration = (i - start) / self.sr
                if duration >= min_silence_duration:
                    silence_regions.append((start/self.sr, i/self.sr))
                start = None
        
        # 통계 계산
        total_duration = len(self.y) / self.sr
        speech_duration = total_duration - sum(end-start for start, end in silence_regions)
        pause_count = len(silence_regions)
        
        results = {
            'speech_rate': speech_duration / total_duration,  # 발화 비율
            'pause_frequency': pause_count / total_duration,  # 초당 휴지 횟수
            'average_pause_duration': sum(end-start for start, end in silence_regions) / max(1, pause_count),
            'pause_regions': silence_regions
        }
        
        return results
    
    def analyze_pitch_variation(self, time_step=0.01):
        """
        음성 주파수(피치) 변화 분석
        :param time_step: 피치 추출 간격 (초)
        """
        # Praat을 사용한 피치 추출
        pitch = call(self.sound, "To Pitch", time_step, 75, 500)
        pitch_values = pitch.selected_array['frequency']
        
        # 유효한 피치값만 선택 (0이 아닌 값)
        valid_pitch = pitch_values[pitch_values != 0]
        
        if len(valid_pitch) == 0:
            return None
        
        # 통계 계산
        pitch_stats = {
            'mean': float(np.mean(valid_pitch)),
            'std': float(np.std(valid_pitch)),
            'range': float(np.ptp(valid_pitch)),
            'coefficient_of_variation': float(np.std(valid_pitch) / np.mean(valid_pitch)),
            'pitch_values': valid_pitch.tolist()
        }
        
        return pitch_stats
    
    def plot_analysis(self, output_path='voice_analysis.png'):
        """모든 분석 결과를 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 1. 파형 및 RMS 에너지
        plt.subplot(4, 1, 1)
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.5)
        plt.title('Waveform and Energy')
        
        # 2. 스펙트로그램
        plt.subplot(4, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # 3. 피치 변화
        pitch_data = self.analyze_pitch_variation()
        if pitch_data:
            plt.subplot(4, 1, 3)
            plt.plot(np.linspace(0, len(self.y)/self.sr, len(pitch_data['pitch_values'])), 
                    pitch_data['pitch_values'])
            plt.title('Pitch Contour')
            plt.ylabel('Frequency (Hz)')
        
        # 4. 휴지 구간
        pause_data = self.analyze_speech_rate_and_pauses()
        plt.subplot(4, 1, 4)
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.5)
        for start, end in pause_data['pause_regions']:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.title('Pause Regions (highlighted in red)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    # 사용 예시
    input_path = "DATA/DePaul/HyangYoung.mov"  # 비디오 또는 오디오 파일 경로
    analyzer = VoiceAnalyzer(input_path)
    
    # 1. 음성 긴장도 분석
    tension = analyzer.analyze_voice_tension()
    print("\n음성 긴장도 분석 결과:")
    print(f"스펙트럴 센트로이드: {tension['spectral_centroid']:.2f}")
    print(f"스펙트럴 롤오프: {tension['spectral_rolloff']:.2f}")
    print(f"에너지: {tension['energy']:.2f}")
    
    # 2. 말하기 속도 및 휴지 분석
    speech_rate = analyzer.analyze_speech_rate_and_pauses()
    print("\n말하기 속도 및 휴지 분석 결과:")
    print(f"발화 비율: {speech_rate['speech_rate']:.2f}")
    print(f"초당 휴지 횟수: {speech_rate['pause_frequency']:.2f}")
    print(f"평균 휴지 길이: {speech_rate['average_pause_duration']:.2f}초")
    
    # 3. 피치 변화 분석
    pitch_variation = analyzer.analyze_pitch_variation()
    if pitch_variation:
        print("\n피치 변화 분석 결과:")
        print(f"평균 피치: {pitch_variation['mean']:.2f} Hz")
        print(f"피치 표준편차: {pitch_variation['std']:.2f} Hz")
        print(f"피치 범위: {pitch_variation['range']:.2f} Hz")
        print(f"변동 계수: {pitch_variation['coefficient_of_variation']:.2f}")
    
    # 4. 시각화
    analyzer.plot_analysis()
    print("\n✅ 분석 결과가 'voice_analysis.png'로 저장되었습니다.")

if __name__ == "__main__":
    main() 