import streamlit as st
import cv2
import numpy as np
import tempfile

# ==========================================
# IMPORT MEDIAPIPE
# ==========================================
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"⚠️ Błąd krytyczny MediaPipe: {e}")

# ==========================================
# INTERFEJS
# ==========================================
st.title("🏐 Analiza Skoku AI")

if mp_pose is None:
    st.warning("Aplikacja nie może wystartować bez biblioteki MediaPipe.")
else:
    st.success("MediaPipe załadowane pomyślnie!")

    uploaded_file = st.file_uploader(
        "Wgraj nagranie swojego skoku (MP4, MOV)",
        type=["mp4", "mov"]
    )

    if uploaded_file:

        # Zapis pliku tymczasowego
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            frame_id = 0  # do kontrolowania FPS w Cloud
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_id += 1

                # 🔹 Przetwarzaj co 2 klatkę, aby Cloud nie crashował
                if frame_id % 2 != 0:
                    continue

                # Zmniejszenie rozdzielczości (wydajność w Streamlit Cloud)
                frame = cv2.resize(frame, (640, 360))

                # Konwersja do RGB dla MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # Rysowanie punktów
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                # 🔥 Wyświetlanie w Streamlit (tylko RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame)

        cap.release()
