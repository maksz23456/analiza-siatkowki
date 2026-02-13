import streamlit as st
import cv2
import numpy as np
import os

# ==========================================
# PANCERNY IMPORT MEDIAPIPE
# ==========================================
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp
    # Próba dostępu do modułów w sposób bezpośredni
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    else:
        # Alternatywna ścieżka dla problematycznych instalacji
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.drawing_utils as mp_drawing
except Exception as e:
    st.error(f"⚠️ Błąd krytyczny MediaPipe: {e}")
    st.info("Twoja wersja Pythona (3.14) może nie być wspierana. Spróbuj zainstalować: python -m pip install mediapipe")

# ==========================================
# RESZTA KODU
# ==========================================
st.title("🏐 Analiza Skoku AI")

if mp_pose is None:
    st.warning("Aplikacja nie może wystartować bez biblioteki MediaPipe.")
else:
    st.success("MediaPipe załadowane pomyślnie!")
    # Tutaj wstaw resztę logiki z VolleyballAnalyzer
  # Panel do wgrywania filmu
    uploaded_file = st.file_uploader("Wgraj nagranie swojego skoku (MP4, MOV)", type=['mp4', 'mov'])

    if uploaded_file:
        # Prowizoryczne zapisanie pliku, aby OpenCV mógł go odczytać
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # Inicjalizacja modelu Pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Konwersja kolorów dla MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Rysowanie punktów (nadgarstki 15, 16 oraz kostki 27, 28)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Wyświetlanie przetworzonej klatki w Streamlit
                stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
