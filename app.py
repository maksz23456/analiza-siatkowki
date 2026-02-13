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
# FUNKCJE POMOCNICZE
# ==========================================
def rotate_image(image, angle):
    """Obraca obraz o zadany kąt"""
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# ==========================================
# INTERFEJS
# ==========================================
st.title("🏐 Analiza Skoku AI")

# Opcja rotacji
rotation = st.selectbox(
    "Obrót wideo (jeśli film jest przekrzywiony)",
    [0, 90, 180, 270],
    index=1  # domyślnie 90 stopni
)

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
    
    # Dane do analizy
    max_height = 0
    jump_detected = False
    
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
            
            # 🔹 Obrót jeśli potrzebny
            if rotation > 0:
                frame = rotate_image(frame, rotation)
            
            # 🔹 Zmniejszenie rozdzielczości
            frame = cv2.resize(frame, (480, 270))
            h, w = frame.shape[:2]
            
            # 🔹 Konwersja do RGB dla MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # 🔹 Rysowanie punktów i analiza
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Pobierz wysokość bioder (środek ciała)
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_y = (left_hip.y + right_hip.y) / 2
                
                # Zapisz maksymalną wysokość (najmniejsze Y = najwyżej)
                if hip_y < max_height or max_height == 0:
                    max_height = hip_y
                    jump_detected = True
                
                # Rysowanie szkieletu
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                # Wyświetl informację o skoku
                jump_height_px = int((1 - max_height) * h)
                cv2.putText(frame, f"Max wysokosc: {jump_height_px}px", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 🔹 Aktualizacja progress bar
            progress = frame_id / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Przetwarzanie: {int(progress*100)}%")
            
            # 🔹 Wyświetlanie co 5 przetworzoną klatkę
            if processed % 5 == 0:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Gotowe!")
    
    cap.release()
    
    # 🔹 Podsumowanie
    if jump_detected:
        st.success("🎉 Analiza zakończona!")
        st.metric("🏐 Maksymalna wysokość skoku", f"{int((1-max_height)*270)} pikseli")
        st.info("💡 Wyższy wynik = wyższy skok! (skala w pikselach)")
    else:
        st.warning("⚠️ Nie wykryto skoku. Upewnij się, że osoba jest dobrze widoczna.")
