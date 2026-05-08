import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import tempfile
import os
from streamlit_cropper import st_cropper
from PIL import Image

st.set_page_config(page_title="물벼룩 BPM 분석", layout="wide")

st.title("🔬 물벼룩 심장박동(BPM) 분석기- 용인홍천고 융합과학동아리 반타")

# 1. 파일 업로드
uploaded_file = st.file_uploader("분석할 동영상 파일을 업로드하세요", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Streamlit은 업로드된 파일을 메모리에 저장하므로, OpenCV에서 읽기 위해 임시 파일로 저장합니다.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        st.error("영상 FPS를 읽을 수 없습니다.")
    else:
        ret, first_frame = cap.read()
        if not ret:
            st.error("영상을 불러올 수 없습니다.")
        else:
            h_img, w_img = first_frame.shape[:2]
            
            st.subheader("1. 심장 부위(ROI) 영역 설정")
            st.write("아래 이미지에서 밝게 표시된 네모 박스를 마우스로 드래그하거나 크기를 조절하여 심장 부위에 맞추세요.")
            
            # 첫 프레임을 PIL 이미지로 변환
            pil_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            
            # st_cropper 내부의 강제 축소(700px 제한)를 풀고, 가로 1200px 크기로 시원하게 렌더링합니다.
            display_scale = 1.0
            target_width = 1200
            if w_img < target_width:
                display_scale = target_width / w_img
                new_w = int(w_img * display_scale)
                new_h = int(h_img * display_scale)
                pil_img_display = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                pil_img_display = pil_img
            
            # 마우스 드래그가 가능한 Cropper 위젯 생성 (should_resize_image=False 가 핵심)
            box = st_cropper(
                pil_img_display, 
                realtime_update=True, 
                box_color='#00FF00', 
                aspect_ratio=None, 
                return_type='box',
                should_resize_image=False
            )
            
            # 크로퍼에서 반환된 확대된 좌표를 다시 정확히 원본 비율로 축소
            x = int(box['left'] / display_scale)
            y = int(box['top'] / display_scale)
            w = int(box['width'] / display_scale)
            h = int(box['height'] / display_scale)
            
            st.write(f"선택된 영역: 가로 시작({x}), 세로 시작({y}), 폭({w}), 높이({h})")
            

            
            st.subheader("2. 분석 실행")
            if st.button("🚀 BPM 분석 시작"):
                with st.spinner("영상을 분석 중입니다... (영상 길이에 따라 시간이 소요될 수 있습니다)"):
                    signals = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        heart_region = gray[y:y+h, x:x+w]
                        
                        # 심장 부위 평균 밝기 기록
                        signals.append(np.mean(heart_region))
                        
                    cap.release()
                    
                    # --- 신호 처리 알고리즘 (기존 코드와 동일) ---
                    signals = np.array(signals)
                    time = np.arange(len(signals)) / fps
                    
                    signals_norm = (signals - np.mean(signals)) / np.std(signals)
                    
                    window = int(fps * 0.3)
                    if window % 2 == 0: window += 1
                    if window < 5: window = 5
                    
                    if len(signals_norm) > window:
                        smooth = savgol_filter(signals_norm, window_length=window, polyorder=2)
                    else:
                        smooth = signals_norm
                        
                    min_distance = int(fps * 0.12)
                    prominence = 0.3
                    
                    peaks, _ = find_peaks(smooth, distance=min_distance, prominence=prominence)
                    peaks_inv, _ = find_peaks(-smooth, distance=min_distance, prominence=prominence)
                    
                    if len(peaks_inv) > len(peaks):
                        final_signal = -smooth
                        final_peaks = peaks_inv
                    else:
                        final_signal = smooth
                        final_peaks = peaks
                        
                    duration_sec = len(signals) / fps
                    heart_rate_bpm = len(final_peaks) / duration_sec * 60
                    
                    # --- 결과 출력 ---
                    st.success("✅ 분석이 완료되었습니다!")
                    
                    # 4개의 지표를 예쁘게 나란히 배치
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("영상 FPS", f"{fps:.2f}")
                    m2.metric("측정 시간", f"{duration_sec:.2f}초")
                    m3.metric("감지된 박동 수", f"{len(final_peaks)}회")
                    m4.metric("♥️ 심박수 (BPM)", f"{heart_rate_bpm:.1f}")
                    
                    # 그래프 그리기
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(time, final_signal, label="Heart signal")
                    ax.plot(time[final_peaks], final_signal[final_peaks], "ro", label="Detected beats")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Normalized brightness change")
                    ax.set_title(f"Daphnia Heart Rate: {heart_rate_bpm:.1f} BPM")
                    ax.legend()
                    fig.tight_layout()
                    
                    # Streamlit으로 그래프 출력
                    st.pyplot(fig)
                    
    # 분석 후 생성된 임시 파일 삭제
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass
