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
    stframe = st.empty()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_id = 0  # do kontroli liczby klatek (wydajność)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1

            # 🔹 Przetwarzaj co 2 klatkę, żeby Cloud się nie zawiesił
            if frame_id % 2 != 0:
                continue

            # 🔹 Zmniejszenie rozdzielczości dla szybszego wyświetlania
            frame = cv2.resize(frame, (640, 360))

            # 🔹 Konwersja do RGB dla MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 🔹 Rysowanie punktów
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # 🔹 Wyświetlanie w Streamlit
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
