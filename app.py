import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from collections import deque
import math

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

def detect_ball(frame):
    """Wykrywa białą piłkę na obrazie"""
    # Konwersja do HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Zakres dla białego koloru (piłka siatkarska)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Maska białego koloru
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Znajdź największy kontur (prawdopodobnie piłka)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Tylko jeśli kontur jest wystarczająco duży
        if area > 100:  # minimalna wielkość piłki
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                radius = int(math.sqrt(area / math.pi))
                return (cx, cy, radius)
    
    return None

def calculate_distance(p1, p2):
    """Oblicza odległość między dwoma punktami"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# ==========================================
# INTERFEJS
# ==========================================
st.title("🏐 Zaawansowana Analiza Skoku AI")

col1, col2 = st.columns(2)
with col1:
    rotation = st.selectbox(
        "Obrót wideo",
        [0, 90, 180, 270],
        index=1
    )
with col2:
    show_skeleton = st.checkbox("Pokaż szkielet", value=True)

uploaded_file = st.file_uploader(
    "Wgraj nagranie skoku atakującego (MP4, MOV)",
    type=["mp4", "mov"]
)

if uploaded_file:
    # 🔹 Zapis pliku
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Informacje o wideo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"📹 Wideo: {total_frames} klatek, {fps} FPS")
    
    # Interfejs
    progress_bar = st.progress(0)
    status_text = st.empty()
    stframe = st.empty()
    
    # Metryki na żywo
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metric_height = metrics_col1.empty()
    metric_speed = metrics_col2.empty()
    metric_ball = metrics_col3.empty()
    
    # Dane do analizy
    height_data = []
    ball_trajectory = deque(maxlen=30)  # ostatnie 30 pozycji piłki
    max_height = None
    min_hip_y = 1.0  # najmniejsze Y (najwyżej)
    ball_hit_point = None
    ball_land_point = None
    max_ball_speed = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_id = 0
        processed = 0
        prev_ball_pos = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_id += 1
            
            # Przetwarzaj co 2 klatkę
            if frame_id % 2 != 0:
                continue
            
            processed += 1
            
            # Obrót
            if rotation > 0:
                frame = rotate_image(frame, rotation)
            
            # Zachowaj proporcje przy zmniejszaniu
            original_h, original_w = frame.shape[:2]
            target_w = 640
            target_h = int(original_h * (target_w / original_w))
            frame = cv2.resize(frame, (target_w, target_h))
            h, w = frame.shape[:2]
            
            # Konwersja RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # ==========================================
            # ANALIZA POZY ZAWODNIKA
            # ==========================================
            current_height = 0
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Wysokość bioder
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_y = (left_hip.y + right_hip.y) / 2
                
                # Zapisz maksymalną wysokość
                if hip_y < min_hip_y:
                    min_hip_y = hip_y
                
                # Wysokość w pikselach (od dołu)
                current_height = int((1 - hip_y) * h)
                max_height = int((1 - min_hip_y) * h)
                height_data.append(current_height)
                
                # Rysowanie szkieletu
                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                
                # Punkt uderzenia piłki (ręka)
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                hand_x = int(((right_hand.x + left_hand.x) / 2) * w)
                hand_y = int(((right_hand.y + left_hand.y) / 2) * h)
            
            # ==========================================
            # DETEKCJA PIŁKI
            # ==========================================
            ball_info = detect_ball(frame)
            ball_speed = 0
            
            if ball_info:
                bx, by, br = ball_info
                ball_trajectory.append((bx, by))
                
                # Rysuj piłkę
                cv2.circle(frame, (bx, by), br, (0, 255, 255), 3)
                cv2.circle(frame, (bx, by), 3, (0, 0, 255), -1)
                
                # Oblicz prędkość piłki
                if prev_ball_pos:
                    distance = calculate_distance(prev_ball_pos, (bx, by))
                    ball_speed = distance * fps / 2  # piksele/sekundę
                    if ball_speed > max_ball_speed:
                        max_ball_speed = ball_speed
                
                prev_ball_pos = (bx, by)
                
                # Wykryj moment uderzenia (piłka blisko ręki)
                if results.pose_landmarks and ball_hit_point is None:
                    dist_to_hand = calculate_distance((bx, by), (hand_x, hand_y))
                    if dist_to_hand < 50:  # 50 pikseli
                        ball_hit_point = (bx, by)
                
                # Wykryj lądowanie (piłka na dole kadru)
                if by > h * 0.85 and ball_land_point is None:
                    ball_land_point = (bx, by)
            
            # ==========================================
            # RYSOWANIE TRAJEKTORII PIŁKI
            # ==========================================
            if len(ball_trajectory) > 1:
                for i in range(1, len(ball_trajectory)):
                    cv2.line(frame, ball_trajectory[i-1], ball_trajectory[i], 
                            (255, 255, 0), 2)
            
            # Zaznacz punkt uderzenia
            if ball_hit_point:
                cv2.drawMarker(frame, ball_hit_point, (0, 255, 0), 
                              cv2.MARKER_CROSS, 20, 3)
                cv2.putText(frame, "UDERZ.", (ball_hit_point[0]+10, ball_hit_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Zaznacz punkt lądowania
            if ball_land_point:
                cv2.drawMarker(frame, ball_land_point, (0, 0, 255), 
                              cv2.MARKER_CROSS, 20, 3)
                cv2.putText(frame, "LAD.", (ball_land_point[0]+10, ball_land_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # ==========================================
            # WYŚWIETLANIE INFORMACJI
            # ==========================================
            # Panel informacji
            info_y = 30
            cv2.putText(frame, f"Wysokosc: {current_height}px (Max: {max_height}px)", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if ball_speed > 0:
                cv2.putText(frame, f"Predkosc pilki: {int(ball_speed)} px/s", 
                           (10, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Aktualizacja metryk na żywo
            metric_height.metric("🏐 Wysokość skoku", f"{max_height}px")
            metric_speed.metric("⚡ Max prędkość piłki", f"{int(max_ball_speed)} px/s")
            metric_ball.metric("🎯 Status piłki", 
                             "Wykryta ✓" if ball_info else "Szukam...")
            
            # Progress bar
            progress = frame_id / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Przetwarzanie: {int(progress*100)}%")
            
            # Wyświetlanie
            if processed % 3 == 0:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analiza zakończona!")
    
    cap.release()
    
    # ==========================================
    # PODSUMOWANIE
    # ==========================================
    st.success("🎉 Analiza zakończona!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏐 Maksymalna wysokość skoku", f"{max_height} px")
    col2.metric("⚡ Maksymalna prędkość piłki", f"{int(max_ball_speed)} px/s")
    
    if ball_hit_point and ball_land_point:
        distance = int(calculate_distance(ball_hit_point, ball_land_point))
        col3.metric("📏 Dystans uderzenie → lądowanie", f"{distance} px")
    
    # Wykres wysokości w czasie
    if height_data:
        st.subheader("📊 Wysokość skoku w czasie")
        import pandas as pd
        df = pd.DataFrame({
            'Klatka': range(len(height_data)),
            'Wysokość (px)': height_data
        })
        st.line_chart(df.set_index('Klatka'))
    
    # Informacje dodatkowe
    st.info("""
    💡 **Jak interpretować wyniki:**
    - **Wysokość skoku**: wyższy wynik = wyższy skok
    - **Prędkość piłki**: szybkość ruchu piłki w pikselach na sekundę
    - **Trajektoria**: żółta linia pokazuje lot piłki
    - **Punkty**: zielony = uderzenie, czerwony = lądowanie
    """)

