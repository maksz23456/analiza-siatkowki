import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

# ==========================================
# MEDIAPIPE
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# INTERFEJS
# ==========================================
st.title("🏐 Analiza Skoku AI")

uploaded_file = st.file_uploader(
    "Wgraj nagranie swojego skoku (MP4, MOV)",
    type=["mp4", "mov"]
)

if uploaded_file:
    # 🔹 Zapis tymczasowego pliku wideo
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Pobierz informacje o wideo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"📹 Wideo: {total_frames} klatek, {fps} FPS")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    stframe = st.empty()
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_id = 0
        processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_id += 1
            
            # 🔹 Przetwarzaj co 3 klatkę (szybciej)
            if frame_id % 3 != 0:
                continue
            
            processed += 1
            
            # 🔹 Zmniejszenie rozdzielczości
            frame = cv2.resize(frame, (480, 270))
            
            # 🔹 Konwersja do RGB dla MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # 🔹 Rysowanie punktów
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
            
            # 🔹 Aktualizacja progress bar
            progress = frame_id / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Przetwarzanie: {int(progress*100)}%")
            
            # 🔹 Wyświetlanie co 5 przetworzoną klatkę (jeszcze szybciej)
            if processed % 5 == 0:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        progress_bar.progress(100)
        status_text.text("✅ Gotowe!")
    
    cap.release()
    st.success("🎉 Analiza zakończona!")
