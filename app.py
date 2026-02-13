import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import math
import os
import subprocess

st.set_page_config(page_title="Volleyball Analysis", page_icon="🏐", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def rotate_image(image, angle):
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def pixels_to_meters(pixels, reference_height_px, reference_height_m):
    if reference_height_px == 0:
        return 0
    return (pixels / reference_height_px) * reference_height_m

def is_ball_in_court(ball_x, ball_y, frame_width, frame_height):
    court_left = frame_width * 0.15
    court_right = frame_width * 0.85
    court_top = frame_height * 0.6
    return court_left <= ball_x <= court_right and ball_y >= court_top

st.title("🏐 Analiza Skoku Siatkarskiego")

col1, col2, col3 = st.columns(3)
with col1:
    rotation = st.selectbox("Obrót wideo", [0, 90, 180, 270], index=1)
with col2:
    net_height_m = st.number_input("Wysokość siatki (m)", 2.24, 2.43, 2.43, 0.01)
with col3:
    show_skeleton = st.checkbox("Pokaż szkielet", value=True)

uploaded_file = st.file_uploader("Wgraj nagranie (MP4, MOV)", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"📹 Wideo: {total_frames} klatek, {fps} FPS, Długość: {total_frames/fps:.1f}s")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    output_path = tempfile.mktemp(suffix='.mp4')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    
    if rotation > 0:
        first_frame = rotate_image(first_frame, rotation)
    
    original_h, original_w = first_frame.shape[:2]
    target_w = 640
    target_h = int(original_h * (target_w / original_w))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    height_data = []
    min_hip_y = 1.0
    player_height_px = None
    ball_hit_point = None
    ball_land_point = None
    ball_in_court = None
    max_ball_speed = 0
    all_ball_positions = []
    current_ball_speed = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_id = 0
        prev_ball_pos = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            if rotation > 0:
                frame = rotate_image(frame, rotation)
            
            frame = cv2.resize(frame, (target_w, target_h))
            h, w = frame.shape[:2]
            
            display_frame = frame.copy()
            
            should_analyze = (frame_id % 2 == 0)
            
            current_height_m = 0
            ball_info = None
            
            if should_analyze:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    
                    hip_y = (left_hip.y + right_hip.y) / 2
                    ankle_y = (left_ankle.y + right_ankle.y) / 2
                    
                    net_y_normalized = 0.20
                    net_y_px = int(net_y_normalized * h)
                    
                    if player_height_px is None:
                        floor_y_normalized = ankle_y
                        reference_px = int((floor_y_normalized - net_y_normalized) * h)
                        player_height_px = reference_px
                    
                    jump_height_from_net_px = int((net_y_normalized - hip_y) * h)
                    current_height_m = pixels_to_meters(jump_height_from_net_px, player_height_px, net_height_m)
                    
                    if hip_y < min_hip_y:
                        min_hip_y = hip_y
                    
                    height_data.append({'frame': frame_id, 'height_m': current_height_m})
                    
                    cv2.line(display_frame, (0, net_y_px), (w, net_y_px), (255, 0, 255), 3)
                    cv2.putText(display_frame, f"SIATKA {net_height_m}m", 
                               (10, net_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    if show_skeleton:
                        mp_drawing.draw_landmarks(
                            display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                        )
                    
                    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    hand_x = int(((right_hand.x + left_hand.x) / 2) * w)
                    hand_y = int(((right_hand.y + left_hand.y) / 2) * h)
                
                ball_info = detect_ball(frame)
                
                if ball_info:
                    bx, by, br = ball_info
                    all_ball_positions.append((bx, by))
                    
                    if prev_ball_pos:
                        distance = calculate_distance(prev_ball_pos, (bx, by))
                        current_ball_speed = distance * fps
                        if current_ball_speed > max_ball_speed:
                            max_ball_speed = current_ball_speed
                    
                    prev_ball_pos = (bx, by)
                    
                    if results.pose_landmarks and ball_hit_point is None:
                        dist_to_hand = calculate_distance((bx, by), (hand_x, hand_y))
                        if dist_to_hand < 60:
                            ball_hit_point = (bx, by)
                    
                    if by > h * 0.80 and ball_land_point is None:
                        ball_land_point = (bx, by)
                        ball_in_court = is_ball_in_court(bx, by, w, h)
            
            # RYSUJ TRAJEKTORIĘ
            if len(all_ball_positions) > 1:
                for i in range(1, len(all_ball_positions)):
                    cv2.line(display_frame, all_ball_positions[i-1], all_ball_positions[i], (255, 255, 0), 3)
            
            if ball_info:
                bx, by, br = ball_info
                cv2.circle(display_frame, (bx, by), br, (0, 255, 255), 3)
            
            if ball_hit_point:
                cv2.drawMarker(display_frame, ball_hit_point, (0, 255, 0), cv2.MARKER_STAR, 30, 4)
                cv2.putText(display_frame, "WYSTAWA", (ball_hit_point[0]+15, ball_hit_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if ball_land_point:
                color = (0, 255, 0) if ball_in_court else (0, 0, 255)
                text = "POLE" if ball_in_court else "AUT"
                cv2.drawMarker(display_frame, ball_land_point, color, cv2.MARKER_STAR, 30, 4)
                cv2.putText(display_frame, text, (ball_land_point[0]+15, ball_land_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            court_left = int(w * 0.15)
            court_right = int(w * 0.85)
            court_top = int(h * 0.6)
            cv2.rectangle(display_frame, (court_left, court_top), (court_right, h), (0, 255, 0), 2)
            
            # METRYKI NA FILMIE
            y_offset = 30
            if current_height_m > 0:
                cv2.putText(display_frame, f"Wysokosc ataku: {current_height_m:.2f}m", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 40
            
            if current_ball_speed > 0:
                cv2.putText(display_frame, f"Predkosc pilki: {int(current_ball_speed)} px/s", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            out.write(display_frame)
            
            if frame_id % 10 == 0:
                progress = frame_id / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Przetwarzanie: {int(progress*100)}%")
        
        progress_bar.progress(1.0)
        status_text.success("✅ Przetwarzanie zakończone!")
    
    cap.release()
    out.release()
    
    status_text.text("🔄 Konwersja do formatu webowego...")
    output_path_web = output_path.replace('.mp4', '_web.mp4')
    
    try:
        subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 
                       '-crf', '23', '-preset', 'fast', '-y', output_path_web], 
                       check=True, capture_output=True, timeout=300)
        if os.path.exists(output_path_web) and os.path.getsize(output_path_web) > 0:
            output_path = output_path_web
    except:
        pass
    
    if height_data:
        max_height_m = max([d['height_m'] for d in height_data])
    else:
        max_height_m = 0
    
    status_text.empty()
    progress_bar.empty()
    
    st.success("✅ Analiza zakończona!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏐 Max wysokość ataku", f"{max_height_m:.2f} m")
    col2.metric("⚡ Max prędkość piłki", f"{int(max_ball_speed)} px/s")
    
    if ball_land_point:
        status = "🟢 W POLU" if ball_in_court else "🔴 AUT"
        col3.metric("🎯 Lądowanie", status)
    
    st.subheader("🎬 Wideo z analizą")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.info(f"📹 Rozmiar: {file_size_mb:.2f} MB | Długość: {total_frames/fps:.1f}s")
        
        with open(output_path, 'rb') as video_file:
            video_bytes = video_file.read()
        
        st.video(video_bytes)
        
        st.download_button("⬇️ Pobierz wideo", video_bytes, f"analiza_{uploaded_file.name}", 
                          "video/mp4", use_container_width=True)
    else:
        st.error("Błąd zapisu wideo")
    
    if height_data:
        st.subheader("📊 Wykres wysokości")
        import pandas as pd
        df = pd.DataFrame(height_data)
        df['Czas (s)'] = df['frame'] / fps
        chart_data = pd.DataFrame({'Czas (s)': df['Czas (s)'], 'Wysokość (m)': df['height_m']})
        st.line_chart(chart_data.set_index('Czas (s)'))


