import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import math
import json
import os
from datetime import datetime
from pathlib import Path

# ==========================================
# KONFIGURACJA
# ==========================================
st.set_page_config(
    page_title="Volleyball Pro Analytics",
    page_icon="🏐",
    layout="wide"
)

# Folder na dane
DATA_DIR = Path("/tmp/volleyball_data")
DATA_DIR.mkdir(exist_ok=True)

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
    """Sprawdza czy piłka wylądowała w polu czy aucie"""
    court_left = frame_width * 0.15
    court_right = frame_width * 0.85
    court_top = frame_height * 0.6
    
    if court_left <= ball_x <= court_right and ball_y >= court_top:
        return True
    return False

def save_analysis_to_db(player_name, action_type, metrics, video_file):
    """Zapisuje analizę do prostej bazy danych (JSON)"""
    analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "player_name": player_name,
        "action_type": action_type,
        "metrics": metrics,
        "video_filename": video_file
    }
    
    # Zapisz do pliku JSON
    db_file = DATA_DIR / f"{analysis_id}.json"
    with open(db_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return analysis_id

def load_all_analyses():
    """Wczytuje wszystkie analizy z bazy"""
    analyses = []
    for json_file in DATA_DIR.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                analyses.append(data)
        except:
            pass
    
    # Sortuj po dacie (najnowsze pierwsze)
    analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return analyses

# ==========================================
# SIDEBAR - NAWIGACJA
# ==========================================
st.sidebar.title("🏐 Volleyball Pro Analytics")
page = st.sidebar.radio(
    "Nawigacja",
    ["📊 Nowa Analiza", "📁 Historia Analiz", "👥 Porównaj Zawodników", "⚙️ Ustawienia"]
)

# ==========================================
# STRONA: NOWA ANALIZA
# ==========================================
if page == "📊 Nowa Analiza":
    st.title("📊 Nowa Analiza Akcji")
    
    # Formularz danych zawodnika
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Imię i nazwisko zawodnika", placeholder="np. Jan Kowalski")
    with col2:
        action_type = st.selectbox(
            "Typ akcji",
            ["Atak", "Blok", "Serwis", "Przyjęcie", "Wystawienie"]
        )
    
    # Kalibracja
    st.subheader("⚙️ Kalibracja")
    col1, col2 = st.columns(2)
    
    with col1:
        calibration_method = st.radio(
            "Metoda kalibracji:",
            ["Wysokość siatki", "Wzrost zawodnika"]
        )
    
    with col2:
        if calibration_method == "Wysokość siatki":
            net_height_m = st.number_input(
                "Wysokość siatki (m)", 
                min_value=2.24, 
                max_value=2.43, 
                value=2.43,
                step=0.01
            )
            reference_height_m = net_height_m
        else:
            player_height_m = st.number_input(
                "Wzrost zawodnika (m)", 
                min_value=1.5, 
                max_value=2.3, 
                value=1.85,
                step=0.01
            )
            reference_height_m = player_height_m
    
    # Ustawienia wideo
    col1, col2 = st.columns(2)
    with col1:
        rotation = st.selectbox("Obrót wideo", [0, 90, 180, 270], index=1)
    with col2:
        show_skeleton = st.checkbox("Pokaż szkielet", value=True)
    
    # Upload wideo
    st.subheader("📹 Wgraj nagranie")
    uploaded_file = st.file_uploader(
        "Wybierz plik wideo (MP4, MOV)",
        type=["mp4", "mov"]
    )
    
    if uploaded_file and player_name:
        # Zapisz wideo tymczasowo
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"📹 Wideo: {total_frames} klatek, {fps} FPS")
        
        # Przycisk przetwarzania
        if st.button("🚀 Rozpocznij Analizę", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Kontenery na wyniki
            results_container = st.container()
            
            with st.spinner("Analizuję wideo..."):
                # Dane analizy
                height_data = []
                min_hip_y = 1.0
                player_height_px = None
                ball_hit_point = None
                ball_land_point = None
                ball_in_court = None
                max_ball_speed = 0
                all_ball_positions = []
                
                # Przygotuj wyjściowe wideo
                output_path = tempfile.mktemp(suffix='.mp4')
                
                # KLUCZOWE: Użyj właściwego kodeka
                # Najpierw sprawdź wymiary pierwszej klatki
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, first_frame = cap.read()
                
                if rotation > 0:
                    first_frame = rotate_image(first_frame, rotation)
                
                original_h, original_w = first_frame.shape[:2]
                target_w = 640
                target_h = int(original_h * (target_w / original_w))
                
                # VideoWriter z mp4v (najprostszy, najbardziej kompatybilny)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
                
                # Reset do początku
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as pose:
                    frame_id = 0
                    prev_ball_pos = None
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_id += 1
                        
                        # Przetwarzaj co 2 klatkę
                        if frame_id % 2 != 0:
                            continue
                        
                        # Obrót
                        if rotation > 0:
                            frame = rotate_image(frame, rotation)
                        
                        # Resize
                        original_h, original_w = frame.shape[:2]
                        target_w = 640
                        target_h = int(original_h * (target_w / original_w))
                        frame = cv2.resize(frame, (target_w, target_h))
                        h, w = frame.shape[:2]
                        
                        display_frame = frame.copy()
                        
                        # Analiza pozy
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)
                        
                        current_height_m = 0
                        
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            nose = landmarks[mp_pose.PoseLandmark.NOSE]
                            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                            
                            hip_y = (left_hip.y + right_hip.y) / 2
                            ankle_y = (left_ankle.y + right_ankle.y) / 2
                            
                            # KALIBRACJA
                            if calibration_method == "Wysokość siatki":
                                net_y_normalized = 0.20
                                net_y_px = int(net_y_normalized * h)
                                
                                if player_height_px is None:
                                    floor_y_normalized = ankle_y
                                    reference_px = int((floor_y_normalized - net_y_normalized) * h)
                                    player_height_px = reference_px
                                
                                jump_height_from_net_px = int((net_y_normalized - hip_y) * h)
                                current_height_m = pixels_to_meters(jump_height_from_net_px, player_height_px, reference_height_m)
                                
                                # Rysuj siatka
                                cv2.line(display_frame, (0, net_y_px), (w, net_y_px), (255, 0, 255), 3)
                                cv2.putText(display_frame, f"SIATKA {reference_height_m}m", 
                                           (10, net_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            else:
                                if player_height_px is None:
                                    player_height_px = int((ankle_y - nose.y) * h)
                                
                                floor_y = ankle_y
                                hip_height_from_floor_px = int((floor_y - hip_y) * h)
                                current_height_m = pixels_to_meters(hip_height_from_floor_px, player_height_px, reference_height_m)
                            
                            if hip_y < min_hip_y:
                                min_hip_y = hip_y
                            
                            height_data.append({
                                'frame': frame_id,
                                'height_m': current_height_m
                            })
                            
                            # Szkielet
                            if show_skeleton:
                                mp_drawing.draw_landmarks(
                                    display_frame,
                                    results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                                )
                            
                            # Ręka
                            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                            left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                            hand_x = int(((right_hand.x + left_hand.x) / 2) * w)
                            hand_y = int(((right_hand.y + left_hand.y) / 2) * h)
                        
                        # Detekcja piłki
                        ball_info = detect_ball(frame)
                        ball_speed = 0
                        
                        if ball_info:
                            bx, by, br = ball_info
                            all_ball_positions.append((bx, by))
                            
                            if prev_ball_pos:
                                distance = calculate_distance(prev_ball_pos, (bx, by))
                                ball_speed = distance * fps
                                if ball_speed > max_ball_speed:
                                    max_ball_speed = ball_speed
                            
                            prev_ball_pos = (bx, by)
                            
                            if results.pose_landmarks and ball_hit_point is None:
                                dist_to_hand = calculate_distance((bx, by), (hand_x, hand_y))
                                if dist_to_hand < 60:
                                    ball_hit_point = (bx, by)
                            
                            if by > h * 0.80 and ball_land_point is None:
                                ball_land_point = (bx, by)
                                ball_in_court = is_ball_in_court(bx, by, w, h)
                        
                        # Rysuj trajektorię
                        if len(all_ball_positions) > 1:
                            for i in range(1, len(all_ball_positions)):
                                cv2.line(display_frame, all_ball_positions[i-1], all_ball_positions[i], 
                                        (255, 255, 0), 3)
                        
                        if ball_info:
                            cv2.circle(display_frame, (bx, by), br, (0, 255, 255), 3)
                        
                        if ball_hit_point:
                            cv2.drawMarker(display_frame, ball_hit_point, (0, 255, 0), 
                                          cv2.MARKER_STAR, 30, 4)
                        
                        if ball_land_point:
                            color = (0, 255, 0) if ball_in_court else (0, 0, 255)
                            cv2.drawMarker(display_frame, ball_land_point, color, 
                                          cv2.MARKER_STAR, 30, 4)
                        
                        # Pole
                        court_left = int(w * 0.15)
                        court_right = int(w * 0.85)
                        court_top = int(h * 0.6)
                        cv2.rectangle(display_frame, (court_left, court_top), (court_right, h), 
                                     (0, 255, 0), 2)
                        
                        # Info
                        cv2.putText(display_frame, f"Wysokosc: {current_height_m:.2f}m", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Zapisz klatkę do wideo
                        out.write(display_frame)
                        
                        # Progress
                        progress = frame_id / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Przetwarzanie: {int(progress*100)}%")
                    
                    progress_bar.progress(1.0)
                    status_text.success("✅ Analiza zakończona!")
                
                cap.release()
                out.release()  # WAŻNE: zamknij VideoWriter
                
                # KONWERSJA DO FORMATU KOMPATYBILNEGO Z PRZEGLĄDARKĄ
                # Użyj ffmpeg do re-encodowania
                output_path_web = output_path.replace('.mp4', '_web.mp4')
                
                import subprocess
                try:
                    # ffmpeg konwersja do H.264 z yuv420p (najbardziej kompatybilne)
                    subprocess.run([
                        'ffmpeg', '-i', output_path,
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-y',  # overwrite
                        output_path_web
                    ], check=True, capture_output=True)
                    
                    # Użyj przekonwertowanego wideo
                    if os.path.exists(output_path_web) and os.path.getsize(output_path_web) > 0:
                        output_path = output_path_web
                except Exception as e:
                    st.warning(f"Nie udało się użyć ffmpeg, próbuję oryginalnego pliku. Error: {e}")
                
                # Oblicz metryki
                if height_data:
                    max_height_m = max([d['height_m'] for d in height_data])
                else:
                    max_height_m = 0
                
                metrics = {
                    "max_jump_height_m": round(max_height_m, 2),
                    "max_ball_speed_px_s": int(max_ball_speed),
                    "ball_in_court": ball_in_court,
                    "calibration_method": calibration_method,
                    "reference_height_m": reference_height_m
                }
                
                # Zapisz do bazy
                analysis_id = save_analysis_to_db(
                    player_name=player_name,
                    action_type=action_type,
                    metrics=metrics,
                    video_file=uploaded_file.name
                )
                
                # Wyświetl wyniki
                with results_container:
                    st.success(f"✅ Analiza zapisana! ID: {analysis_id}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🏐 Maksymalna wysokość", f"{max_height_m:.2f} m")
                    col2.metric("⚡ Max prędkość piłki", f"{int(max_ball_speed)} px/s")
                    
                    if ball_land_point:
                        status = "🟢 W POLU" if ball_in_court else "🔴 AUT"
                        col3.metric("🎯 Lądowanie", status)
                    
                    # ODTWARZANIE PEŁNEGO WIDEO
                    st.subheader("🎬 Wideo z analizą")
                    
                    # Sprawdź czy plik istnieje i ma rozmiar
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        # Wyświetl informacje o pliku
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        st.info(f"📹 Plik wideo: {file_size_mb:.2f} MB")
                        
                        # Odczytaj plik
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        # Próbuj wyświetlić
                        try:
                            st.video(video_bytes)
                        except Exception as e:
                            st.error(f"Nie można wyświetlić wideo: {e}")
                            st.info("💡 Użyj przycisku poniżej aby pobrać wideo i obejrzeć lokalnie")
                        
                        # ZAWSZE daj opcję pobrania
                        st.download_button(
                            label="⬇️ Pobierz wideo z analizą",
                            data=video_bytes,
                            file_name=f"{player_name}_{action_type}_{analysis_id}.mp4",
                            mime="video/mp4"
                        )
                    else:
                        st.error("⚠️ Nie udało się zapisać wideo. Spróbuj ponownie.")
                    
                    # Wykres
                    if height_data:
                        st.subheader("📊 Wykres wysokości")
                        import pandas as pd
                        df = pd.DataFrame(height_data)
                        df['Czas (s)'] = df['frame'] / fps
                        chart_data = pd.DataFrame({
                            'Czas (s)': df['Czas (s)'],
                            'Wysokość (m)': df['height_m']
                        })
                        st.line_chart(chart_data.set_index('Czas (s)'))
    
    elif uploaded_file and not player_name:
        st.warning("⚠️ Podaj imię i nazwisko zawodnika")

# ==========================================
# STRONA: HISTORIA ANALIZ
# ==========================================
elif page == "📁 Historia Analiz":
    st.title("📁 Historia Analiz")
    
    analyses = load_all_analyses()
    
    if not analyses:
        st.info("Brak zapisanych analiz. Wykonaj pierwszą analizę!")
    else:
        st.write(f"**Znaleziono {len(analyses)} analiz**")
        
        # Tabela wyników
        import pandas as pd
        
        table_data = []
        for a in analyses:
            table_data.append({
                "ID": a['id'],
                "Data": datetime.fromisoformat(a['timestamp']).strftime("%Y-%m-%d %H:%M"),
                "Zawodnik": a['player_name'],
                "Akcja": a['action_type'],
                "Wyskok (m)": a['metrics'].get('max_jump_height_m', 0),
                "Prędkość piłki": a['metrics'].get('max_ball_speed_px_s', 0),
                "Status": "🟢 Pole" if a['metrics'].get('ball_in_court') else "🔴 Aut" if a['metrics'].get('ball_in_court') is not None else "-"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Szczegóły wybranej analizy
        st.subheader("🔍 Szczegóły analizy")
        selected_id = st.selectbox("Wybierz analizę", [a['id'] for a in analyses])
        
        if selected_id:
            selected = next(a for a in analyses if a['id'] == selected_id)
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(selected['metrics'])
            with col2:
                st.write(f"**Zawodnik:** {selected['player_name']}")
                st.write(f"**Akcja:** {selected['action_type']}")
                st.write(f"**Plik:** {selected['video_filename']}")

# ==========================================
# STRONA: PORÓWNAJ ZAWODNIKÓW
# ==========================================
elif page == "👥 Porównaj Zawodników":
    st.title("👥 Porównanie Zawodników")
    
    analyses = load_all_analyses()
    
    if len(analyses) < 2:
        st.warning("⚠️ Potrzebujesz co najmniej 2 analizy aby porównać zawodników!")
    else:
        st.write("Wybierz dwie analizy do porównania:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Zawodnik A")
            analysis_a_id = st.selectbox(
                "Wybierz analizę A",
                [f"{a['player_name']} - {a['action_type']} ({a['id']})" for a in analyses],
                key="player_a"
            )
        
        with col2:
            st.subheader("👤 Zawodnik B")
            analysis_b_id = st.selectbox(
                "Wybierz analizę B",
                [f"{a['player_name']} - {a['action_type']} ({a['id']})" for a in analyses],
                key="player_b"
            )
        
        if st.button("📊 Porównaj", type="primary"):
            # Pobierz dane
            a_id = analysis_a_id.split("(")[-1].strip(")")
            b_id = analysis_b_id.split("(")[-1].strip(")")
            
            analysis_a = next(a for a in analyses if a['id'] == a_id)
            analysis_b = next(a for a in analyses if a['id'] == b_id)
            
            st.subheader("📊 Wyniki porównania")
            
            # Porównanie metryk
            col1, col2, col3 = st.columns(3)
            
            height_a = analysis_a['metrics'].get('max_jump_height_m', 0)
            height_b = analysis_b['metrics'].get('max_jump_height_m', 0)
            
            with col1:
                st.metric(
                    "🏐 Wyskok - " + analysis_a['player_name'],
                    f"{height_a:.2f} m",
                    delta=f"{height_a - height_b:+.2f} m" if height_b else None
                )
            
            with col2:
                st.metric(
                    "🏐 Wyskok - " + analysis_b['player_name'],
                    f"{height_b:.2f} m",
                    delta=f"{height_b - height_a:+.2f} m" if height_a else None
                )
            
            with col3:
                winner = analysis_a['player_name'] if height_a > height_b else analysis_b['player_name']
                st.success(f"🏆 Wyższy skok: **{winner}**")
            
            # Wykres porównawczy
            st.subheader("📈 Wykres porównawczy")
            import pandas as pd
            
            comparison_data = pd.DataFrame({
                'Zawodnik': [analysis_a['player_name'], analysis_b['player_name']],
                'Wyskok (m)': [height_a, height_b],
                'Prędkość piłki (px/s)': [
                    analysis_a['metrics'].get('max_ball_speed_px_s', 0),
                    analysis_b['metrics'].get('max_ball_speed_px_s', 0)
                ]
            })
            
            st.bar_chart(comparison_data.set_index('Zawodnik'))

# ==========================================
# STRONA: USTAWIENIA
# ==========================================
elif page == "⚙️ Ustawienia":
    st.title("⚙️ Ustawienia")
    
    st.subheader("📊 Statystyki systemu")
    analyses = load_all_analyses()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba analiz", len(analyses))
    
    unique_players = len(set(a['player_name'] for a in analyses))
    col2.metric("Liczba zawodników", unique_players)
    
    col3.metric("Rozmiar bazy", f"{len(list(DATA_DIR.glob('*.json')))} plików")
    
    st.subheader("🗑️ Zarządzanie danymi")
    
    if st.button("🗑️ Wyczyść wszystkie dane", type="secondary"):
        for json_file in DATA_DIR.glob("*.json"):
            json_file.unlink()
        st.success("✅ Wszystkie dane zostały usunięte!")
        st.experimental_rerun()
    
    st.subheader("ℹ️ O aplikacji")
    st.info("""
    **Volleyball Pro Analytics** v1.0
    
    Profesjonalne narzędzie do analizy siatkarskiej:
    - ✅ Analiza skoków i akcji
    - ✅ Pomiar wysokości w metrach
    - ✅ Detekcja piłki i trajektoria
    - ✅ Baza danych analiz
    - ✅ Porównanie zawodników
    
    🚀 W przygotowaniu:
    - Analiza całych meczów
    - Statystyki sezonowe
    - Mapy cieplne uderzeń
    - Export raportów PDF
    """)
