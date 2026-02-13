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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Zakres dla białego koloru
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 80:
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

def pixels_to_meters(pixels, reference_height_px, reference_height_m):
    """Konwertuje piksele na metry"""
    if reference_height_px == 0:
        return 0
    return (pixels / reference_height_px) * reference_height_m

def is_ball_in_court(ball_x, ball_y, frame_width, frame_height):
    """
    Sprawdza czy piłka wylądowała w polu czy aucie
    Zakładamy że pole to środkowa część kadru
    """
    # Definiujemy pole jako środkowe 70% szerokości i dolne 40% wysokości
    court_left = frame_width * 0.15
    court_right = frame_width * 0.85
    court_top = frame_height * 0.6
    
    if court_left <= ball_x <= court_right and ball_y >= court_top:
        return True  # W polu
    return False  # Aut

# ==========================================
# INTERFEJS
# ==========================================
st.title("🏐 Profesjonalna Analiza Skoku Siatkarskiego")

# Ustawienia kalibracji
st.sidebar.header("⚙️ Kalibracja")
player_height_m = st.sidebar.number_input(
    "Wzrost zawodnika (metry)", 
    min_value=1.5, 
    max_value=2.3, 
    value=1.85,
    step=0.01
)

st.sidebar.info("""
💡 **Wzrost zawodnika** służy do kalibracji:
- Wpisz prawdziwy wzrost zawodnika
- System automatycznie przeliczy piksele na metry
""")

col1, col2 = st.columns(2)
with col1:
    rotation = st.selectbox("Obrót wideo", [0, 90, 180, 270], index=1)
with col2:
    show_skeleton = st.checkbox("Pokaż szkielet", value=True)

uploaded_file = st.file_uploader(
    "Wgraj nagranie skoku atakującego (MP4, MOV)",
    type=["mp4", "mov"]
)

if uploaded_file:
    # Zapis pliku
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"📹 Wideo: {total_frames} klatek, {fps} FPS")
    
    # ==========================================
    # PRZETWARZANIE WIDEO I ZAPISYWANIE KLATEK
    # ==========================================
    if 'processed_frames' not in st.session_state:
        st.write("🔄 Przetwarzanie wideo...")
        progress_bar = st.progress(0)
        
        frames_data = []
        ball_positions = []
        height_data = []
        
        min_hip_y = 1.0
        player_height_px = None
        ball_hit_point = None
        ball_land_point = None
        ball_in_court = None
        max_ball_speed = 0
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            frame_id = 0
            prev_ball_pos = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                frame_id += 1
                
                # Przetwarzaj co 2 klatkę
                if frame_id % 2 != 0:
                    continue
                
                # Obrót
                if rotation > 0:
                    frame = rotate_image(frame, rotation)
                
                # Zmień rozmiar zachowując proporcje
                original_h, original_w = frame.shape[:2]
                target_w = 640
                target_h = int(original_h * (target_w / original_w))
                frame = cv2.resize(frame, (target_w, target_h))
                h, w = frame.shape[:2]
                
                # Kopia do zapisu
                display_frame = frame.copy()
                
                # Analiza pozy
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                current_height_px = 0
                current_height_m = 0
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Wysokość bioder
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    hip_y = (left_hip.y + right_hip.y) / 2
                    
                    # Wysokość głowy i stóp (do kalibracji)
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    ankle_y = (left_ankle.y + right_ankle.y) / 2
                    
                    # Oblicz wysokość zawodnika w pikselach (od stóp do głowy)
                    if player_height_px is None:
                        player_height_px = int((ankle_y - nose.y) * h)
                    
                    # Zapisz maksymalną wysokość
                    if hip_y < min_hip_y:
                        min_hip_y = hip_y
                    
                    # Wysokość w pikselach i metrach
                    current_height_px = int((1 - hip_y) * h)
                    current_height_m = pixels_to_meters(
                        current_height_px, 
                        player_height_px, 
                        player_height_m
                    )
                    
                    height_data.append({
                        'frame': frame_id,
                        'height_px': current_height_px,
                        'height_m': current_height_m
                    })
                    
                    # Rysowanie szkieletu
                    if show_skeleton:
                        mp_drawing.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                        )
                    
                    # Punkt ręki
                    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    hand_x = int(((right_hand.x + left_hand.x) / 2) * w)
                    hand_y = int(((right_hand.y + left_hand.y) / 2) * h)
                
                # Detekcja piłki
                ball_info = detect_ball(frame)
                ball_speed = 0
                
                if ball_info:
                    bx, by, br = ball_info
                    ball_positions.append({'frame': frame_id, 'x': bx, 'y': by, 'r': br})
                    
                    # Oblicz prędkość
                    if prev_ball_pos:
                        distance = calculate_distance(prev_ball_pos, (bx, by))
                        ball_speed = distance * fps / 2
                        if ball_speed > max_ball_speed:
                            max_ball_speed = ball_speed
                    
                    prev_ball_pos = (bx, by)
                    
                    # Wykryj moment uderzenia
                    if results.pose_landmarks and ball_hit_point is None:
                        dist_to_hand = calculate_distance((bx, by), (hand_x, hand_y))
                        if dist_to_hand < 60:
                            ball_hit_point = {'frame': frame_id, 'x': bx, 'y': by}
                    
                    # Wykryj lądowanie
                    if by > h * 0.80 and ball_land_point is None:
                        ball_land_point = {'frame': frame_id, 'x': bx, 'y': by}
                        ball_in_court = is_ball_in_court(bx, by, w, h)
                else:
                    ball_positions.append(None)
                
                # Zapisz klatkę
                frames_data.append({
                    'frame': display_frame.copy(),
                    'height_px': current_height_px,
                    'height_m': current_height_m,
                    'ball_speed': ball_speed
                })
                
                # Progress
                progress = frame_id / total_frames
                progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
        
        cap.release()
        
        # Zapisz do session_state
        st.session_state.processed_frames = frames_data
        st.session_state.ball_positions = ball_positions
        st.session_state.height_data = height_data
        st.session_state.player_height_px = player_height_px
        st.session_state.ball_hit_point = ball_hit_point
        st.session_state.ball_land_point = ball_land_point
        st.session_state.ball_in_court = ball_in_court
        st.session_state.max_ball_speed = max_ball_speed
        st.session_state.min_hip_y = min_hip_y
        st.session_state.fps = fps
        
        st.experimental_rerun()
    
    # ==========================================
    # ODTWARZANIE Z KONTROLKAMI
    # ==========================================
    if 'processed_frames' in st.session_state:
        frames_data = st.session_state.processed_frames
        ball_positions = st.session_state.ball_positions
        height_data = st.session_state.height_data
        player_height_px = st.session_state.player_height_px
        ball_hit_point = st.session_state.ball_hit_point
        ball_land_point = st.session_state.ball_land_point
        ball_in_court = st.session_state.ball_in_court
        max_ball_speed = st.session_state.max_ball_speed
        min_hip_y = st.session_state.min_hip_y
        fps = st.session_state.fps
        
        st.success("✅ Wideo przetworzone!")
        
        # Metryki końcowe
        max_height_px = int((1 - min_hip_y) * frames_data[0]['frame'].shape[0])
        max_height_m = pixels_to_meters(max_height_px, player_height_px, player_height_m)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🏐 Max wysokość skoku", f"{max_height_m:.2f} m ({max_height_px} px)")
        col2.metric("⚡ Max prędkość piłki", f"{int(max_ball_speed)} px/s")
        
        if ball_land_point:
            status = "🟢 W POLU" if ball_in_court else "🔴 AUT"
            col3.metric("🎯 Lądowanie piłki", status)
        
        # Kontrolki odtwarzania
        st.subheader("🎬 Odtwarzanie")
        
        frame_idx = st.slider(
            "Klatka",
            0,
            len(frames_data) - 1,
            0,
            label_visibility="collapsed"
        )
        
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("⏮️ Początek"):
            frame_idx = 0
        if col2.button("◀️ -10 klatek"):
            frame_idx = max(0, frame_idx - 10)
        if col3.button("▶️ +10 klatek"):
            frame_idx = min(len(frames_data) - 1, frame_idx + 10)
        if col4.button("⏭️ Koniec"):
            frame_idx = len(frames_data) - 1
        
        # Pobierz aktualną klatkę
        current_data = frames_data[frame_idx]
        display_frame = current_data['frame'].copy()
        h, w = display_frame.shape[:2]
        
        # Rysuj trajektorię piłki (wszystkie pozycje do tej klatki)
        trajectory_points = []
        for i, ball_pos in enumerate(ball_positions[:frame_idx + 1]):
            if ball_pos:
                trajectory_points.append((ball_pos['x'], ball_pos['y']))
        
        # Rysuj trajektorię
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(display_frame, trajectory_points[i-1], trajectory_points[i], 
                        (255, 255, 0), 3)
        
        # Rysuj aktualną pozycję piłki
        if ball_positions[frame_idx]:
            ball = ball_positions[frame_idx]
            cv2.circle(display_frame, (ball['x'], ball['y']), ball['r'], (0, 255, 255), 3)
            cv2.circle(display_frame, (ball['x'], ball['y']), 3, (0, 0, 255), -1)
        
        # Zaznacz punkt uderzenia
        if ball_hit_point and frame_idx >= ball_hit_point['frame']:
            cv2.drawMarker(display_frame, (ball_hit_point['x'], ball_hit_point['y']), 
                          (0, 255, 0), cv2.MARKER_STAR, 30, 4)
            cv2.putText(display_frame, "UDERZ.", 
                       (ball_hit_point['x']+15, ball_hit_point['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Zaznacz punkt lądowania
        if ball_land_point and frame_idx >= ball_land_point['frame']:
            color = (0, 255, 0) if ball_in_court else (0, 0, 255)
            text = "POLE" if ball_in_court else "AUT"
            cv2.drawMarker(display_frame, (ball_land_point['x'], ball_land_point['y']), 
                          color, cv2.MARKER_STAR, 30, 4)
            cv2.putText(display_frame, text, 
                       (ball_land_point['x']+15, ball_land_point['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Rysuj pole (wizualizacja)
        court_left = int(w * 0.15)
        court_right = int(w * 0.85)
        court_top = int(h * 0.6)
        cv2.rectangle(display_frame, (court_left, court_top), (court_right, h), 
                     (0, 255, 0), 2)
        
        # Info na ekranie
        info_y = 30
        cv2.putText(display_frame, f"Wysokosc: {current_data['height_m']:.2f}m ({current_data['height_px']}px)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if current_data['ball_speed'] > 0:
            cv2.putText(display_frame, f"Predkosc: {int(current_data['ball_speed'])} px/s", 
                       (10, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(display_frame, f"Klatka: {frame_idx + 1}/{len(frames_data)}", 
                   (10, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Wyświetl
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Wykres wysokości
        st.subheader("📊 Wykres wysokości skoku w czasie")
        import pandas as pd
        df = pd.DataFrame(height_data)
        df['Czas (s)'] = df['frame'] / fps
        
        chart_data = pd.DataFrame({
            'Czas (s)': df['Czas (s)'],
            'Wysokość (m)': df['height_m']
        })
        st.line_chart(chart_data.set_index('Czas (s)'))
        
        # Przycisk do ponownego przetworzenia
        if st.button("🔄 Przetwórz ponownie"):
            del st.session_state.processed_frames
            st.experimental_rerun()
