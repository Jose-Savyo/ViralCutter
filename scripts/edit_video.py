import cv2
import numpy as np
import os
import subprocess
import mediapipe as mp
from scripts.one_face import crop_and_resize_single_face, resize_with_padding, detect_face_or_body, crop_center_zoom, compose_fit_blur, compose_fit_faces, subpixel_crop_resize
from scripts.virtual_camera import CameraRig, crop_window, CAMERA_STYLES
from scripts.two_face import crop_and_resize_two_faces, detect_face_or_body_two_faces
from scripts.stage_timer import ClipTimer, save_clip_timings
try:
    from scripts.face_detection_insightface import init_insightface, detect_faces_insightface, crop_and_resize_insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not found or error importing. Install with: pip install insightface onnxruntime-gpu")


# Global cache for encoder
CACHED_ENCODER = None
CLIP_TIMINGS = []  # tempos por corte da ultima chamada de edit()

def _encoder_works(encoder_name, preset):
    """Codifica alguns quadros de teste: o ffmpeg listar o encoder nao garante que exista o hardware."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=256x256:d=0.2',
           '-c:v', encoder_name, '-preset', preset, '-pix_fmt', 'yuv420p', '-f', 'null', '-']
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False

def get_best_encoder():
    global CACHED_ENCODER
    if CACHED_ENCODER: return CACHED_ENCODER

    try:
        # Check available encoders
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
        output = result.stdout

        # Priority: NVENC (NVIDIA) > AMF (AMD) > QSV (Intel) > VideoToolbox (Mac) > CPU
        candidates = [
            ("h264_nvenc", "fast", "NVIDIA"),
            ("h264_amf", "speed", "AMD"),
            ("h264_qsv", "veryfast", "Intel QSV"),
            ("h264_videotoolbox", "default", "MacOS"),
        ]
        for name, preset, label in candidates:
            if name in output:
                if _encoder_works(name, preset):
                    print(f"Encoder Detected: {label} ({name})")
                    CACHED_ENCODER = (name, preset)
                    return CACHED_ENCODER
                print(f"Encoder {name} listed by ffmpeg but not usable on this machine. Skipping.")

    except Exception as e:
        print(f"Error checking encoders: {e}")

    print("Encoder Detected: CPU (libx264)")
    CACHED_ENCODER = ("libx264", "ultrafast")
    return CACHED_ENCODER


class FFmpegPipeWriter:
    """
    Grava quadros BGR direto em H.264 pelo pipe do ffmpeg, com o melhor encoder disponivel.
    Substitui o cv2.VideoWriter('mp4v'): o mp4v era um encode extra na CPU, com perda, e
    obrigava a juncao com o audio a re-encodar o video. Agora a juncao so copia o video.
    """
    def __init__(self, path, fps, width=1080, height=1920):
        self.size = (width, height)
        self._warned = False
        encoder_name, encoder_preset = get_best_encoder()
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-hide_banner',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
               '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
               '-c:v', encoder_name, '-preset', encoder_preset, '-pix_fmt', 'yuv420p']
        if encoder_name == 'libx264':
            cmd += ['-crf', '18']      # intermediario: a legenda re-encoda depois
        else:
            cmd += ['-b:v', '8M']
        cmd.append(path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError, ValueError) as e:
            if not self._warned:
                print(f"Error writing frame to ffmpeg pipe: {e}")
                self._warned = True

    def release(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait()

FRAME_LAYOUTS = ("crop", "fit-blur", "fit-faces")

# Correlacao de histograma abaixo deste valor entre quadros consecutivos = troca de cena/camera.
# Calibrado num podcast de estudio: quadros seguidos ~1.0, fade ~0.88, corte seco entre cameras
# do mesmo estudio (mesmas cores) ~0.72.
SCENE_CUT_THRESHOLD = 0.80

def frame_histogram(frame):
    """Histograma HSV (H x S) normalizado de uma versao reduzida do quadro. Barato o bastante para todo quadro."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def is_scene_cut(prev_hist, cur_hist, threshold=SCENE_CUT_THRESHOLD):
    if prev_hist is None or cur_hist is None:
        return False
    return cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_CORREL) < threshold

# --- fit-faces: rostos persistentes e layout adaptativo ---
STABLE_ZONE_RATIO = 0.11      # camera parada enquanto o rosto anda menos que 11% da altura do quadro (~120 px em 1080p)
REFRAME_SECONDS = 0.5         # duracao do movimento quando precisa reenquadrar
FIT_BIG_FACE_RATIO = 0.18      # rosto com altura > 18% do quadro = camera de estudio -> usa crop
FIT_SWITCH_SECONDS = 0.75      # tempo minimo antes de trocar crop <-> fit-faces (fora troca de cena)
FIT_CONTENT_FACE_FACTOR = 1.6  # com 3+ rostos, maior > 1.6x o segundo = conteudo da tela, nao webcam


class FacePersistence:
    """
    Conta em quantas deteccoes cada regiao do quadro teve rosto. Webcams de transmissao ficam
    sempre no mesmo lugar; rostos do jogo/video aparecem, somem e se movem.
    """
    def __init__(self, cols=16, rows=9, min_rounds=6):
        self.cols, self.rows, self.min_rounds = cols, rows, min_rounds
        self.hits = np.zeros((rows, cols), dtype=np.float32)
        self.rounds = 0

    def _cell(self, center, fw, fh):
        cx = min(self.cols - 1, max(0, int(center[0] / fw * self.cols)))
        cy = min(self.rows - 1, max(0, int(center[1] / fh * self.rows)))
        return cy, cx

    def update(self, faces, fw, fh):
        self.rounds += 1
        seen = set(self._cell(f['center'], fw, fh) for f in faces)
        for cy, cx in seen:
            self.hits[cy, cx] += 1

    def score(self, center, fw, fh):
        if self.rounds < self.min_rounds:
            return 0.0
        cy, cx = self._cell(center, fw, fh)
        y0, y1 = max(0, cy - 1), min(self.rows, cy + 2)
        x0, x1 = max(0, cx - 1), min(self.cols, cx + 2)
        return float(min(1.0, self.hits[y0:y1, x0:x1].max() / self.rounds))


def face_height_ratio(bbox, frame_h):
    return max(0.0, float(bbox[3]) - float(bbox[1])) / max(1.0, float(frame_h))


def render_no_face(frame, frame_layout="crop", no_face_mode="padding"):
    """Quadro sem rosto utilizavel, respeitando o layout escolhido."""
    if frame_layout == "fit-blur":
        return compose_fit_blur(frame)
    if frame_layout == "fit-faces":
        return compose_fit_faces(frame, [])
    if no_face_mode == "zoom":
        return crop_center_zoom(frame)
    return resize_with_padding(frame)

def get_center_bbox(bbox):
    # bbox: [x1, y1, x2, y2]
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def get_center_rect(rect):
    # rect: (x, y, w, h)
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)

def sort_by_proximity(new_faces, old_faces, center_func):
    """
    Sorts new_faces to match the order of old_faces based on distance.
    new_faces: list of face objects (bbox or tuple)
    old_faces: list of face objects (bbox or tuple)
    center_func: function that takes a face object and returns (cx, cy)
    """
    if not old_faces or len(old_faces) != 2 or len(new_faces) != 2:
        return new_faces
    
    old_c1 = center_func(old_faces[0])
    old_c2 = center_func(old_faces[1])
    
    new_c1 = center_func(new_faces[0])
    new_c2 = center_func(new_faces[1])
    
    # Cost if we keep order: [new1, new2]
    # dist(old1, new1) + dist(old2, new2)
    dist_keep = ((old_c1[0]-new_c1[0])**2 + (old_c1[1]-new_c1[1])**2) + \
                ((old_c2[0]-new_c2[0])**2 + (old_c2[1]-new_c2[1])**2)
                
    # Cost if we swap: [new2, new1]
    # dist(old1, new2) + dist(old2, new1)
    dist_swap = ((old_c1[0]-new_c2[0])**2 + (old_c1[1]-new_c2[1])**2) + \
                ((old_c2[0]-new_c1[0])**2 + (old_c2[1]-new_c1[1])**2)
                
    # If swapping reduces total movement distance, do it
    if dist_swap < dist_keep:
        return [new_faces[1], new_faces[0]]
    
    return new_faces

def generate_short_fallback(input_file, output_file, index, project_folder, final_folder, no_face_mode="padding", frame_layout="crop"):
    """Fallback function: Center Crop (Zoom) or Padding if detection fails. Also the fast path for frame_layout='fit-blur'."""
    print(f"Processing (Fallback): {input_file} | Mode: {no_face_mode} | Layout: {frame_layout}")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ct = ClipTimer(f"Corte {index}", (total_frames / fps) if fps else 0)
    
    # Target dimensions (9:16)
    
    target_width = 1080
    target_height = 1920
    
    encoder_name, encoder_preset = get_best_encoder()
    
    # Use FFmpeg Pipe instead of cv2.VideoWriter to avoid OpenCV backend errors
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error', '-hide_banner', '-stats',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{target_width}x{target_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', encoder_name,
        '-preset', encoder_preset,
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    # If using hardware encoder, we might want to set bitrate to ensure quality
    if "nvenc" in encoder_name or "amf" in encoder_name:
         ffmpeg_cmd.extend(["-b:v", "5M"])
    
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        with ct.measure("leitura"):
            ret, frame = cap.read()
        if not ret:
            break

        result = render_no_face(frame, frame_layout, no_face_mode)

        try:
            # Write raw bytes to ffmpeg stdin
            with ct.measure("escrita"):
                process.stdin.write(result.tobytes())
        except Exception as e:
            print(f"Error writing frame to ffmpeg pipe: {e}")
            pass
        


    cap.release()
    process.stdin.close()
    process.wait()
    
    with ct.measure("mux"):
        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)
    CLIP_TIMINGS.append(ct.finish())

def finalize_video(input_file, output_file, index, fps, project_folder, final_folder):
    """Mux audio and video."""
    audio_file = os.path.join(project_folder, "cuts", f"output-audio-{index}.aac")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_file, "-vn", "-acodec", "copy", audio_file], 
                   check=False, capture_output=True)

    if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
        final_output = os.path.join(final_folder, f"final-output{str(index).zfill(3)}_processed.mp4")
        # O video ja sai em H.264 do enquadramento: so copia (sem re-encode)
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", output_file,
            "-i", audio_file,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_output
        ]
        try:
            subprocess.run(command, check=True) #, capture_output=True)
            print(f"Final file generated: {final_output}")
            try:
                os.remove(audio_file)
                os.remove(output_file) 
            except:
                pass
        except subprocess.CalledProcessError as e:
            print(f"Error muxing: {e}")
    else:
        print(f"Warning: No audio extracted for {input_file}")


def calculate_mouth_ratio(landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR) using 68-point landmarks (inner lips).
    Indices: 
    Inner Lips: 60-67 (0-indexed 60 to 67)
    Left Corner: 60
    Right Corner: 64
    Top Center: 62
    Bottom Center: 66
    """
    if landmarks is None:
        return 0
    
    # 3D points (x,y,z) or 2D (x,y). We use first 2 cols.
    pts = landmarks.astype(float)
    
    # Simple vertical vs horizontal
    # Vertical
    p62 = pts[62]
    p66 = pts[66]
    h = np.linalg.norm(p62[:2] - p66[:2])
    
    # Horizontal
    p60 = pts[60]
    p64 = pts[64]
    w = np.linalg.norm(p60[:2] - p64[:2])
    
    if w < 1e-6: return 0
    
    return h / w

def generate_short_mediapipe(input_file, output_file, index, face_mode, project_folder, final_folder, face_detection, face_mesh, pose, detection_period=None, no_face_mode="padding"):
    try:
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"Error opening video: {input_file}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = FFmpegPipeWriter(output_file, fps)

        next_detection_frame = 0
        current_interval = int(5 * fps) # Initial guess

        # Initial Interval Logic if predefined

        if detection_period is not None:
             current_interval = max(1, int(detection_period * fps))
        elif face_mode == "2":
             current_interval = int(1.0 * fps)
        
        last_detected_faces = None
        last_frame_face_positions = None
        last_success_frame = -1000
        max_frames_without_detection = int(3.0 * fps) # 3 seconds timeout

        transition_duration = int(fps)
        transition_frames = []

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_index >= next_detection_frame:
                # Detect ALL faces (up to 2 in our implementation)
                detections = detect_face_or_body_two_faces(frame, face_detection, face_mesh, pose)
                
                # Dynamic Logic
                target_faces = 1
                if face_mode == "2":
                    target_faces = 2
                elif face_mode == "auto":
                    if detections and len(detections) >= 2:
                        target_faces = 2
                    else:
                        target_faces = 1
                
                # Filter detections based on target
                current_detections = []
                if detections:
                    # Sort detections by approximate Area (w*h) descending to pick main faces first
                    detections.sort(key=lambda s: s[2] * s[3], reverse=True)
                    
                    if len(detections) >= target_faces:
                        current_detections = detections[:target_faces]
                    elif len(detections) > 0:
                        # Fallback
                        current_detections = detections[:1] 
                        target_faces = 1 
                    
                    # Apply Consistency Check (Proximity)
                    if target_faces == 2 and len(current_detections) == 2:
                         if last_detected_faces is not None and len(last_detected_faces) == 2:
                             current_detections = sort_by_proximity(current_detections, last_detected_faces, get_center_rect)
                
                # Check for stability/lookahead could go here but skipping for brevity unless requested.
                
                if current_detections and len(current_detections) == target_faces:
                    if last_frame_face_positions is not None:
                        start_faces = np.array(last_frame_face_positions)
                        end_faces = np.array(current_detections)
                        try:
                            transition_frames = np.linspace(start_faces, end_faces, transition_duration, dtype=int)
                        except Exception as e:
                            # Fallback if shapes mismatch unexpectedly
                            transition_frames = []
                    else:
                        transition_frames = []
                    last_detected_faces = current_detections
                    last_success_frame = frame_index
                else:
                    pass
                
                # Update next detection frame
                step = 5
                
                if detection_period is not None:
                    if isinstance(detection_period, dict):
                         # If we are targeting 2 faces, we use '2' interval, else '1'
                         key = str(target_faces)
                         val = detection_period.get(key, detection_period.get('1', 0.2))
                         step = max(1, int(val * fps))
                    else:
                         step = max(1, int(detection_period * fps))
                elif target_faces == 2:
                    step = int(1.0 * fps)
                else:
                    step = int(5) # 5 frames for 1 face
                
                next_detection_frame = frame_index + step

            if len(transition_frames) > 0:
                current_faces = transition_frames[0]
                transition_frames = transition_frames[1:]
            elif last_detected_faces is not None and (frame_index - last_success_frame) <= max_frames_without_detection:
                current_faces = last_detected_faces
            else:
                if no_face_mode == "zoom":
                    result = crop_center_zoom(frame)
                else:
                    result = resize_with_padding(frame)
                coordinate_log.append({"frame": frame_index, "faces": []})
                out.write(result)
                continue

            last_frame_face_positions = current_faces

            if hasattr(current_faces, '__len__') and len(current_faces) == 2:
                 result = crop_and_resize_two_faces(frame, current_faces)
            else:
                 # Ensure it's list of tuples or single tuple? current_faces is list of tuples from detection
                 # If 1 face: [ (x,y,w,h) ]
                 if hasattr(current_faces, '__len__') and len(current_faces) > 0:
                     f = current_faces[0]
                     result = crop_and_resize_single_face(frame, f)
                 else:
                     if no_face_mode == "zoom":
                         result = crop_center_zoom(frame)
                     else:
                         result = resize_with_padding(frame)
            
            out.write(result)

        cap.release()
        out.release()
        
        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)

    except Exception as e:
        print(f"Error in MediaPipe processing: {e}")
        raise e # Rethrow to trigger fallback

def generate_short_haar(input_file, output_file, index, project_folder, final_folder, detection_period=None, no_face_mode="padding"):
    """Face detection using OpenCV Haar Cascades."""
    print(f"Processing (Haar Cascade): {input_file}")
    
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade XML. Falling back to center crop.")
        generate_short_fallback(input_file, output_file, index, project_folder, final_folder)
        return

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = FFmpegPipeWriter(output_file, fps)
    
    # Logic copied from generate_short_mediapipe
    detection_interval = int(2 * fps) # Default check every 2 seconds
    if detection_period is not None:
        detection_interval = max(1, int(detection_period * fps))
    last_detected_faces = None
    last_frame_face_positions = None
    last_success_frame = -1000
    max_frames_without_detection = int(3.0 * fps)

    transition_duration = int(fps) # 1 second smooth transition
    transition_frames = []

    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_index % detection_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detections = []
            if len(faces) > 0:
                # Pick largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                # Ensure int type
                detections = [tuple(map(int, largest_face))]

            if detections:
                if last_frame_face_positions is not None:
                    # Simple linear interpolation for smoothing
                    start_faces = np.array(last_frame_face_positions)
                    end_faces = np.array(detections)
                    
                    # Generate transition frames
                    steps = transition_duration
                    transition_frames = []
                    for s in range(steps):
                        t = (s + 1) / steps
                        interp = (1 - t) * start_faces + t * end_faces
                        transition_frames.append(interp.astype(int).tolist()) # Convert back to list of lists/tuples
                else:
                    transition_frames = []
                last_detected_faces = detections
                last_success_frame = frame_index
            else:
                pass

        if len(transition_frames) > 0:
            current_faces = transition_frames[0]
            transition_frames = transition_frames[1:]
        elif last_detected_faces is not None and (frame_index - last_success_frame) <= max_frames_without_detection:
            current_faces = last_detected_faces
        else:
            # No face detected for a while -> Center/Padding fallback
            if no_face_mode == "zoom":
                result = crop_center_zoom(frame)
            else:
                result = resize_with_padding(frame)
            out.write(result)
            continue

        last_frame_face_positions = current_faces
        # haar detections are list containing one tuple (x,y,w,h)
        # current_faces is list of one tuple
        if isinstance(current_faces, list):
             face_bbox = current_faces[0]
        else:
             face_bbox = current_faces # Should be handled

        result = crop_and_resize_single_face(frame, face_bbox)
        out.write(result)

    cap.release()
    out.release()

    finalize_video(input_file, output_file, index, fps, project_folder, final_folder)

def generate_short_insightface(input_file, output_file, index, project_folder, final_folder, face_mode="auto", detection_period=None, filter_threshold=0.35, two_face_threshold=0.60, confidence_threshold=0.30, dead_zone=40, focus_active_speaker=False, active_speaker_mar=0.03, active_speaker_score_diff=1.5, include_motion=False, active_speaker_motion_deadzone=3.0, active_speaker_motion_sensitivity=0.05, active_speaker_decay=2.0, no_face_mode="padding", frame_layout="crop", camera_style="normal"):
    """Face detection using InsightFace (SOTA)."""
    print(f"Processing (InsightFace): {input_file} | Mode: {face_mode} | Layout: {frame_layout}")
    
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ct = ClipTimer(f"Corte {index}", (total_frames / fps) if fps else 0)
    
    # Using mp4v for container, but final mux will fix encoding
    out = FFmpegPipeWriter(output_file, fps)
    
    # Dynamic Interval Logic
    next_detection_frame = 0
    
    last_detected_faces = None
    last_frame_face_positions = None
    last_success_frame = -1000
    # Segura a ultima posicao por no maximo 1s sem deteccao (antes 3s: mostrava o rosto
    # errado por segundos depois de uma troca de camera). Troca de cena invalida na hora.
    max_frames_without_detection = int(1.0 * fps)
    prev_hist = None
    scene_cut_count = 0

    # Enquadramento estavel: a camera so se move quando o rosto sai de uma zona central
    # (proporcional ao quadro) e, quando move, desliza em ~0,5 s com suavizacao.
    transition_duration = max(4, int(REFRAME_SECONDS * fps))
    stable_zone = max(float(dead_zone), STABLE_ZONE_RATIO * frame_height)
    transition_frames = []
    # Camera virtual v2 (zona de conforto, confirmacao de salto, mola, corte seco na troca de pessoa)
    rig = CameraRig(frame_width, frame_height, fps, camera_style, dead_zone)

    # Current state of face mode (1 or 2)
    # If auto, we decide per detection interval
    current_num_faces_state = 1
    if face_mode == "2":
        current_num_faces_state = 2

    # No fit-faces as faixas comportam 2 rostos sem dividir a tela: mostra o segundo
    # rosto sempre que ele nao for muito menor que o principal.
    if frame_layout == "fit-faces" and face_mode == "auto":
        two_face_threshold = min(two_face_threshold, 0.30)

    frame_1_face_count = 0
    frame_2_face_count = 0

    buffered_frame = None
    
    # Timeline tracking: list of (frame_index, mode_str)
    # We will compress this later.
    timeline_frames = [] # Store mode for *every written frame* or at least detection points
    
    timeline_frames = [] # Store mode for *every written frame* or at least detection points
    coordinate_log = [] # Store raw face coordinates frame-by-frame
    
    # For Active Speaker Logic
    # Map of "Face ID" to activity score?
    # Since we don't have ID tracker, we blindly assign score to faces based on proximity to previous frame
    # A list of dictionaries: [{'center': (x,y), 'activity': score}, ...]
    faces_activity_state = [] 
    
    # fit-faces: persistencia de rostos + troca crop <-> fit por cena
    face_persistence = FacePersistence()
    fit_sub_layout = None          # "fit" ou "crop"
    fit_candidate, fit_candidate_frames = None, 0
    fit_switch_frames = max(1, int(FIT_SWITCH_SECONDS * fps))
    fit_scene_reset = True
    fit_crop_frames = 0

    for frame_index in range(total_frames):
        if buffered_frame is not None:
             frame = buffered_frame
             ret = True
             buffered_frame = None
        else:
             with ct.measure("leitura"):
                 ret, frame = cap.read()

        if not ret or frame is None:
            break

        # --- SCENE CUT DETECTION ---
        # Na troca de camera/cena: detecta de novo neste quadro e esquece o tracking anterior,
        # sem interpolar entre a posicao antiga e a nova (evita "panoramica" falsa entre cameras).
        cur_hist = frame_histogram(frame)
        if is_scene_cut(prev_hist, cur_hist):
            scene_cut_count += 1
            print(f"DEBUG: Frame {frame_index} | Scene cut -> re-detecting faces")
            next_detection_frame = frame_index
            transition_frames = []
            last_detected_faces = None
            last_frame_face_positions = None
            last_success_frame = -1000
            faces_activity_state = []
            fit_scene_reset = True
            rig.reset(frame_index / fps)
        prev_hist = cur_hist

        if frame_index >= next_detection_frame and len(transition_frames) == 0:
            # Detect faces
            with ct.measure("deteccao"):
                faces = detect_faces_insightface(frame)
            if faces:
                scores = [f"{f.get('det_score',0):.2f}" for f in faces]
                print(f"DEBUG: Frame {frame_index} | Raw Faces: {len(faces)} | Scores: {scores}")
            else:
                pass # print(f"DEBUG: Frame {frame_index} | No Raw Faces")

            # --- ACTIVITY / SPEAKER DETECTION ---
            # (Feature currently disabled for stability - relying on simple size checks)
            last_raw_faces = faces 
            # ------------------------------------

            # --- INTELLIGENT FILTERING ---
            valid_faces = []
            if faces:
                # 1. Filter by confidence (Using user threshold)
                faces = [f for f in faces if f.get('det_score', 0) > confidence_threshold]
                
                if faces:
                    # Pre-calculate areas and SPEAKER SCORE
                    for f in faces:
                        w = f['bbox'][2] - f['bbox'][0]
                        h = f['bbox'][3] - f['bbox'][1]
                        f['area'] = w * h
                        f['center'] = ((f['bbox'][0] + f['bbox'][2]) / 2, (f['bbox'][1] + f['bbox'][3]) / 2)
                        
                        act = f.get('activity', 0)
                        f['effective_area'] = f['area'] * (1.0 + (act * 0.05))

                    if frame_layout == "fit-faces":
                        # Rosto que fica sempre no mesmo lugar (webcam) pesa mais que rosto do jogo/video
                        face_persistence.update(faces, frame_width, frame_height)
                        for f in faces:
                            f['persistence'] = face_persistence.score(f['center'], frame_width, frame_height)
                            f['effective_area'] *= (1.0 + 4.0 * f['persistence'])
                        size_key = 'effective_area'
                        # Tela de transmissao: varias webcams de tamanho parecido + um rosto bem maior
                        # no conteudo (personagem do jogo/video). Esse rosto ja aparece no video central,
                        # entao sai das faixas. Rosto de camera de estudio (grande) nao e afetado.
                        if len(faces) >= 3:
                            by_area = sorted(faces, key=lambda f: f['area'], reverse=True)
                            top, second = by_area[0], by_area[1]
                            if (top['area'] > FIT_CONTENT_FACE_FACTOR * second['area']
                                    and face_height_ratio(top['bbox'], frame_height) <= FIT_BIG_FACE_RATIO):
                                print(f"DEBUG: Frame {frame_index} | fit-faces: ignorando rosto do conteudo da tela ({top['area']/second['area']:.1f}x o segundo)")
                                faces = [f for f in faces if f is not top]
                    else:
                        size_key = 'area'

                    # Find largest face
                    max_area = max(f[size_key] for f in faces)
                    
                    # 2. Relative Size Filter
                    valid_faces = [f for f in faces if f[size_key] > (filter_threshold * max_area)]
                    
                    if len(valid_faces) < len(faces):
                        print(f"DEBUG: Filtered {len(faces)-len(valid_faces)} small faces. Max Area: {max_area}. Filter Thresh: {filter_threshold}")
                    
                    faces = valid_faces
            
            # --- ACTIVE SPEAKER UPDATE ---
            if faces:
                # 1. Update activity scores for current faces
                # Simple matching to previous state
                current_state_map = []
                
                for f in faces:
                    # Calculate instantaneous openness
                    mar = 0
                    if 'landmark_3d_68' in f:
                        mar = calculate_mouth_ratio(f['landmark_3d_68'])
                    elif 'landmark_2d_106' in f:
                        # Fallback or Todo: map 106 to 68 approximate
                        # 106 indices: 52-71 are lips.
                        # Inner roughly 64-71?
                        # Let's rely on 3d_68 which is standard in buffalo_l
                        pass
                    
                    f['mouth_ratio'] = mar
                    # Heuristic: Ratio > 0.05 implies openish, > 0.1 talk.
                    # Adjust thresholds: 0.03 is common for closed mouth, 0.05 is starting to open.
                    
                    # Log raw MAR for debugging
                    # print(f"DEBUG: Frame {frame_index} Face {i} MAR: {mar:.4f}")
                    
                    is_talking = 1.0 if mar > active_speaker_mar else 0.0 
                    

            # --- CROWD MODE LOGIC ---
            # If too many faces, don't even try to track. Fallback to No-Face logic (Zoom/Padding)
            CROWD_THRESHOLD = 7 
            # FIX: Use last_raw_faces (before size filtering) so we count background people too!
            is_crowd = len(last_raw_faces) >= CROWD_THRESHOLD
            if is_crowd:
                print(f"DEBUG: Crowd Mode Active! {len(faces)} faces >= {CROWD_THRESHOLD}. Triggering Fallback (No Face Mode).")
                faces = [] 
                valid_faces = [] # CAUTION: Must clear strict backup too!
                # FORCE RESET HISTORY so it doesn't "stick" to the last face found
                last_detected_faces = None
                transition_frames = []
                faces_activity_state = [] 
                zoom_ema_bbox = None # Reset smoothing too
                rig.reset()
            # ---------------------------

            # Update Activity State - Two Pass for Global Motion Compensation
            if focus_active_speaker and faces:
                # Pass 1: Global Motion (Camera Shake) Calculation
                # We calculate motion for ALL confident faces (before size filtering) to get best global estimate
                raw_motions = []
                
                # First, ensure we have a temporary mapping of current faces to history
                # We do this non-destructively just to get motion values
                for f in faces:
                    my_c = f['center']
                    best_dist = 9999
                    if faces_activity_state:
                         for old_s in faces_activity_state:
                             old_c = old_s['center']
                             dist = np.sqrt((my_c[0]-old_c[0])**2 + (my_c[1]-old_c[1])**2)
                             if dist < best_dist:
                                 best_dist = dist
                    
                    if best_dist < 200:
                        f['_raw_motion'] = best_dist
                    else:
                        f['_raw_motion'] = 0.0
                    
                    if include_motion:
                        raw_motions.append(f['_raw_motion'])

                global_motion = 0.0
                if include_motion and len(raw_motions) >= 2:
                    global_motion = min(raw_motions)

                # Pass 2: Update Scores for ALL faces
                current_state_map = []
                for f in faces:
                     # Helper: Is talking?
                     is_talking = f.get('mouth_ratio', 0) > active_speaker_mar
                     
                     # Calculate Compensated Motion
                     motion_bonus = 0.0
                     if include_motion and faces_activity_state:
                         comp_motion = max(0.0, f.get('_raw_motion', 0.0) - global_motion)
                         f['motion_val'] = comp_motion # Store for debug
                         
                         if comp_motion > active_speaker_motion_deadzone:
                              motion_bonus = min(2.5, (comp_motion - active_speaker_motion_deadzone) * active_speaker_motion_sensitivity)
                     else:
                        f['motion_val'] = 0.0
                     
                     # Accumulate Score
                     matched_score = 0.0
                     
                     # Re-find match to update history
                     my_c = f['center']
                     best_dist = 9999
                     best_idx = -1
                     if faces_activity_state:
                         for i, old_s in enumerate(faces_activity_state):
                             old_c = old_s['center']
                             dist = np.sqrt((my_c[0]-old_c[0])**2 + (my_c[1]-old_c[1])**2)
                             if dist < best_dist:
                                 best_dist = dist
                                 best_idx = i
                     
                     if best_idx != -1 and best_dist < 200:
                         old_val = faces_activity_state[best_idx]['activity']
                         change = -abs(active_speaker_decay)
                         if is_talking:
                             change = 1.5
                         
                         new_val = old_val + change + motion_bonus
                         # Increased cap to 20.0 to allow motion differences to separate two 'talking' faces
                         matched_score = max(0.0, min(20.0, new_val))
                     else:
                         matched_score = 1.0 if is_talking else 0.0
                     
                     f['activity_score'] = matched_score
                     current_state_map.append({'center': f['center'], 'activity': matched_score})
                 
                faces_activity_state = current_state_map
            else:
                faces_activity_state = []

            faces = valid_faces
            
            # Decide 1 or 2 faces
            target_faces = 1
            if face_mode == "2":
                target_faces = 2
            elif face_mode == "auto":
                if len(faces) >= 2:
                    # Default decision variable
                    decided = False
                    
                    if focus_active_speaker:
                         # EXPERIMENTAL: Decide based on activity
                         f1 = faces[0]
                         f2 = faces[1]
                         score1 = f1.get('activity_score', 0)
                         score2 = f2.get('activity_score', 0)
                         
                         y1 = f1['center'][1]
                         y2 = f2['center'][1]
                         pos1 = "Top" if y1 < y2 else "Bottom"
                         pos2 = "Top" if y2 < y1 else "Bottom"
                         
                         # Debug Active Speaker
                         print(f"DEBUG: Frame {frame_index} | {pos1} (MAR: {f1.get('mouth_ratio',0):.3f}, Mov: {f1.get('motion_val',0):.1f}, Score: {score1:.1f}) | {pos2} (MAR: {f2.get('mouth_ratio',0):.3f}, Mov: {f2.get('motion_val',0):.1f}, Score: {score2:.1f})")


                         # If one is clearly dominant active speaker
                         # Lower threshold to make it more sensitive?
                         # Score difference > 2.0 (approx 2-3 frames of talking difference vs silence)
                         diff = abs(score1 - score2)
                         # Check strict dominance first
                         if diff > active_speaker_score_diff:
                             # Pick the winner
                             target_faces = 1
                             decided = True
                             # Ensure the list is sorted by activity so [0] is the winner
                             if score2 > score1:
                                 # Swap ensures [0] is the active one for later 1-face crop logic which takes [0]
                                 faces = [f2, f1]
                             print(f"DEBUG: Active Speaker Focus Triggered! Diff ({diff:.2f}) > Thresh ({active_speaker_score_diff}). Focusing on Face {'2' if score2 > score1 else '1'}.")
                             
                         elif score1 > 4.0 and score2 > 4.0:
                             # Both talking -> 2 faces
                             # Raised threshold to 4.0 to avoid noise triggering split
                             target_faces = 2
                             decided = True
                             print(f"DEBUG: Dual Active Speakers! Both scores > 4.0. Forcing Split Mode.")
                         
                         # If scores are low (both silent), fallback to size ratio (decided=False) or force 1 if very silent?
                         # Let's fallback to size.

                    if not decided:
                        # Standard Logic: Check relative sizes (effective area)
                        faces_sorted_temp = sorted(faces, key=lambda f: f.get('effective_area', 0), reverse=True)
                        largest = faces_sorted_temp[0]['effective_area']
                        second = faces_sorted_temp[1]['effective_area']
    
                        # Two-Face Constraint
                        if second > (two_face_threshold * largest):
                            target_faces = 2
                        else:
                            target_faces = 1
                else:
                    target_faces = 1
            
            # If no faces found effectively after filter
            if not faces and not valid_faces:
                 # Logic ensures faces = valid_faces already
                 pass
            
            # -----------------------------
            
            # Fallback Lookahead: If detection fails or partial
            # But DO NOT look ahead if we are in Crowd Mode (we explicitly wanted 0 faces)
            if len(faces) < target_faces and not is_crowd:
                # Try 1 frame ahead
                with ct.measure("leitura"):
                    ret2, frame2 = cap.read()
                if ret2 and frame2 is not None:
                     with ct.measure("deteccao"):
                         faces2 = detect_faces_insightface(frame2)
                     
                     # --- Apply same filtering to lookahead ---
                     valid_faces2 = []
                     if faces2:
                         faces2 = [f for f in faces2 if f.get('det_score', 0) > confidence_threshold]
                         if faces2:
                             for f in faces2:
                                 w = f['bbox'][2] - f['bbox'][0]
                                 h = f['bbox'][3] - f['bbox'][1]
                                 f['area'] = w * h
                                 f['center'] = ((f['bbox'][0] + f['bbox'][2]) / 2, (f['bbox'][1] + f['bbox'][3]) / 2)
                                 f['effective_area'] = f['area'] # Default for lookahead
                             max_area2 = max(f['area'] for f in faces2)
                             # STRICTER FILTER: threshold of max area
                             valid_faces2 = [f for f in faces2 if f['area'] > (filter_threshold * max_area2)]
                     faces2 = valid_faces2
                     # ----------------------------------------


                     # If lookahead found what we wanted OR found something better than nothing
                     if len(faces2) >= target_faces:
                         faces = faces2 # Use lookahead faces for current frame
                     elif len(faces) == 0 and len(faces2) > 0:
                         faces = faces2 # Better than nothing
                         
                     buffered_frame = frame2 # Store for next iteration

            detections = []
            
            if len(faces) >= target_faces:
                # --- FACE TRACKING / SORTING ---
                # Instead of just Area, we prioritize faces closer to the LAST detected face
                # This prevents switching to a background person if sizes are similar
                
                if last_detected_faces is not None and len(last_detected_faces) == target_faces:
                   # Define score function: High Area is good, Low Distance to old is good.
                   # But simpler: calculate Intersection over Union (IOU) or Distance to old bbox center
                   
                   # We want to match existing slots.
                   # For 1 face:
                   if target_faces == 1:
                       old_center = get_center_bbox(last_detected_faces[0])
                       
                       def sort_score(f):
                           # Distance score (lower is better)
                           dist = np.sqrt((f['center'][0] - old_center[0])**2 + (f['center'][1] - old_center[1])**2)
                           # EFFECTIVE Area score (higher is better)
                           # Weight distance more heavily to keep consistency, but allow activity to swap focus if significant
                           # normalized score?
                           return dist - (f['effective_area'] * 0.0001) 
                       
                       faces_sorted = sorted(faces, key=sort_score)
                   else:
                       # For 2 faces, just sort by effective area for now as proximity sort happens later
                       faces_sorted = sorted(faces, key=lambda f: f['effective_area'], reverse=True)
                else:
                   # No history, sort by effective area
                   if focus_active_speaker and target_faces == 1:
                        # Pick the one with highest activity score
                        faces_sorted = sorted(faces, key=lambda f: f.get('activity_score', 0), reverse=True)
                   else:
                        faces_sorted = sorted(faces, key=lambda f: f.get('effective_area', 0), reverse=True)
                
                if target_faces == 2:
                    # Convert [x1, y1, x2, y2] to (x, y, w, h) logic is later
                    # Ensure we have 2 faces
                    f1 = faces_sorted[0]['bbox']
                    f2 = faces_sorted[1]['bbox']
                    
                    if last_detected_faces is not None and len(last_detected_faces) == 2:
                        detections = sort_by_proximity([f1, f2], last_detected_faces, get_center_bbox)
                    else:
                        detections = [f1, f2]
                        
                    current_num_faces_state = 2
                else:
                    # 1 face
                    detections = [faces_sorted[0]['bbox']]
                    current_num_faces_state = 1
            else:
                 # If we wanted 2 but found 1, or wanted 1 found 0
                 if len(faces) > 0:
                     # Fallback to 1 face if found at least 1
                     faces_sorted = sorted(faces, key=lambda f: f['effective_area'], reverse=True)
                     detections = [faces_sorted[0]['bbox']]
                     current_num_faces_state = 1
                 else:
                     detections = []

            if detections:
                t_now = frame_index / fps
                if (frame_index - last_success_frame) > max_frames_without_detection:
                    rig.reset()  # rosto perdido por muito tempo: volta com corte seco
                rig.observe(detections, t_now)
                last_detected_faces = detections
                last_success_frame = frame_index


            # Update next detection frame based on NEW state
            step = 5 # Default fallback (very fast)
            
            if detection_period is not None:
                if isinstance(detection_period, dict):
                    # Period depends on state
                    key = str(current_num_faces_state) 
                    # fallback to '1' if key not found (should be there)
                    val = detection_period.get(key, detection_period.get('1', 0.2)) 
                    step = max(1, int(val * fps))
                else:
                    # Legacy float support (should not happen with new main.py but good safety)
                    step = max(1, int(detection_period * fps))
            elif current_num_faces_state == 2:
                step = int(1.0 * fps) # 1s for 2 faces
            else:
                step = 5 # 5 frames for 1 face (~0.16s at 30fps)
            
            next_detection_frame = frame_index + step

        if rig.count and (frame_index - last_success_frame) <= max_frames_without_detection:
            current_faces = rig.step(1.0 / fps)
        else:
            # Fallback for this frame
            result = render_no_face(frame, frame_layout, no_face_mode)
            with ct.measure("escrita"):
                out.write(result)
            timeline_frames.append((frame_index, "1")) # Fix: Ensure fallback is treated as single face for subs
            
            # Fix XML Log sync (Empty faces for fallback)
            coords_entry = {"frame": frame_index, "src_size": [frame_width, frame_height], "faces": [], "layout": "nenhum"}
            coordinate_log.append(coords_entry)
            
            continue

        last_frame_face_positions = current_faces
        
        target_len = len(current_faces)

        use_fit = False
        crop_log = None
        if frame_layout == "fit-faces":
             # Cena de camera unica (rosto grande) -> crop normal; tela com webcams pequenas -> fit-faces.
             big = face_height_ratio(current_faces[0], frame_height) > FIT_BIG_FACE_RATIO
             wanted = "crop" if big else "fit"
             if fit_sub_layout is None or fit_scene_reset:
                 fit_sub_layout = wanted
                 fit_scene_reset = False
                 fit_candidate, fit_candidate_frames = None, 0
             elif wanted != fit_sub_layout:
                 if wanted == fit_candidate:
                     fit_candidate_frames += 1
                 else:
                     fit_candidate, fit_candidate_frames = wanted, 1
                 if fit_candidate_frames >= fit_switch_frames:
                     fit_sub_layout = wanted
                     fit_candidate, fit_candidate_frames = None, 0
             else:
                 fit_candidate, fit_candidate_frames = None, 0
             use_fit = fit_sub_layout == "fit"
             if not use_fit:
                 fit_crop_frames += 1
                 # No crop, so divide a tela se os dois rostos forem grandes
                 if target_len == 2 and face_height_ratio(current_faces[1], frame_height) <= FIT_BIG_FACE_RATIO * 0.6:
                     current_faces = [current_faces[0]]
                     target_len = 1

        if use_fit:
             # Video inteiro no centro + zoom nos rostos nas faixas de cima/baixo.
             # Legenda na posicao padrao (modo "1"), logo abaixo do video central.
             if target_len == 2:
                 frame_2_face_count += 1
             else:
                 frame_1_face_count += 1
             result = compose_fit_faces(frame, current_faces)
             timeline_frames.append((frame_index, "1"))
        elif target_len == 2:
             frame_2_face_count += 1
             # Convert [x1, y1, x2, y2] to (x, y, w, h)
             f1 = current_faces[0]
             f2 = current_faces[1]
             rect1 = (f1[0], f1[1], f1[2]-f1[0], f1[3]-f1[1])
             rect2 = (f2[0], f2[1], f2[2]-f2[0], f2[3]-f2[1])
             result = crop_and_resize_two_faces(frame, [rect1, rect2])
             timeline_frames.append((frame_index, "2"))
        else:
             frame_1_face_count += 1
             # 1 face
             # current_faces[0] is [x1, y1, x2, y2]
             fx1, fy1, fx2, fy2 = [float(v) for v in current_faces[0][:4]]
             crop_log = crop_window(frame_width, frame_height, (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)
             result = subpixel_crop_resize(frame, *crop_log, 1080, 1920)
             timeline_frames.append((frame_index, "1"))
             
        # Capture Coordinates (Frame-by-Frame)
        coords_entry = {"frame": frame_index, "src_size": [frame_width, frame_height], "faces": [],
                        "layout": "fit" if use_fit else ("crop2" if target_len == 2 else "crop1")}
        if crop_log is not None:
            coords_entry["crop"] = [round(float(v), 1) for v in crop_log]
        try:
            # We want to store [x1, y1, x2, y2, rh] for each face
            if isinstance(current_faces, (list, tuple)):
                processed_faces_log = []
                for f in current_faces:
                    f_list = list(map(int, f[:4])) # Standard bbox
                    # Calculate rh (relative height)
                    face_h = f_list[3] - f_list[1]
                    rh = face_h / float(frame_height)
                    f_list.append(float(f"{rh:.4f}")) # Append as 5th element
                    processed_faces_log.append(f_list)
                coords_entry["faces"] = processed_faces_log
                
            elif isinstance(current_faces, np.ndarray):
                # Similar logic for numpy
                processed_faces_log = []
                for f in current_faces:
                    f_list = f[:4].astype(int).tolist()
                    face_h = f_list[3] - f_list[1]
                    rh = face_h / float(frame_height)
                    f_list.append(float(f"{rh:.4f}"))
                    processed_faces_log.append(f_list)
                coords_entry["faces"] = processed_faces_log
        except: pass
        coordinate_log.append(coords_entry)

        with ct.measure("escrita"):
            out.write(result)

    cap.release()
    with ct.measure("escrita"):
        out.release()
    
    # Compress timeline into segments
    # [(start_time, end_time, mode), ...]
    compressed_timeline = []
    if timeline_frames:
        curr_mode = timeline_frames[0][1]
        start_f = timeline_frames[0][0]
        
        for i in range(1, len(timeline_frames)):
            frame_idx, mode = timeline_frames[i]
            if mode != curr_mode:
                # End current segment
                # Convert frame to seconds
                end_f = timeline_frames[i-1][0]
                compressed_timeline.append({
                    "start": float(start_f) / fps,
                    "end": float(end_f) / fps, # or frame_idx / fps for continuity
                    "mode": curr_mode
                })
                # Start new
                curr_mode = mode
                start_f = frame_idx
        
        # Add last
        end_f = timeline_frames[-1][0]
        compressed_timeline.append({
             "start": float(start_f) / fps,
             "end": (float(end_f) + 1) / fps,
             "mode": curr_mode
        })
    
    # Save timeline JSON
    timeline_file = output_file.replace(".mp4", "_timeline.json")
    try:
        import json
        with open(timeline_file, "w") as f:
            json.dump(compressed_timeline, f)
        print(f"Timeline saved: {timeline_file}")
    except Exception as e:
        print(f"Error saving timeline: {e}")

    # Save Coords JSON
    coords_file = output_file.replace(".mp4", "_coords.json")
    try:
        with open(coords_file, "w") as f:
            json.dump(coordinate_log, f)
        print(f"Face Coordinates saved: {coords_file}")
    except Exception as e:
        print(f"Error saving coords: {e}")

    print(f"Summary: {scene_cut_count} scene cut(s) | 1-face frames: {frame_1_face_count} | 2-face frames: {frame_2_face_count} | fit-faces->crop frames: {fit_crop_frames} | total: {total_frames}")
    print(f"Camera ({rig.style}): {rig.stats['panoramicas']} panoramica(s), {rig.stats['cortes_pessoa']} corte(s) de pessoa, {rig.stats['trocas_layout']} troca(s) de layout")
    with ct.measure("mux"):
        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)
    CLIP_TIMINGS.append(ct.finish())

    # Return dominant mode logic (or keep 15% rule as overall fallback)
    if frame_2_face_count > (total_frames * 0.15):
        return "2"
    return "1"


def edit(project_folder="tmp", face_model="insightface", face_mode="auto", detection_period=None, filter_threshold=0.35, two_face_threshold=0.60, confidence_threshold=0.30, dead_zone=10, focus_active_speaker=False, active_speaker_mar=0.03, active_speaker_score_diff=1.5, include_motion=False, active_speaker_motion_deadzone=3.0, active_speaker_motion_sensitivity=0.05, active_speaker_decay=2.0, segments_data=None, no_face_mode="padding", frame_layout="crop", camera_style="normal"):
    CLIP_TIMINGS.clear()
    if frame_layout not in FRAME_LAYOUTS:
        print(f"WARNING: Unknown frame_layout '{frame_layout}'. Using 'crop'.")
        frame_layout = "crop"
    print(f"Frame layout: {frame_layout}")
    # fit-blur nao usa rostos: nao precisa carregar detector nenhum
    if frame_layout == "fit-blur":
        face_model = "none"
    # Lazy init solutions only when needed to avoid AttributeError if import failed partially
    mp_face_detection = None
    mp_face_mesh = None
    mp_pose = None
    
    index = 0
    cuts_folder = os.path.join(project_folder, "cuts")
    final_folder = os.path.join(project_folder, "final")
    os.makedirs(final_folder, exist_ok=True)
    
    face_modes_log = {}
    
    # Priority: User Choice -> Fallbacks
    
    insightface_working = False
    
    # Only init InsightFace if selected or default
    if INSIGHTFACE_AVAILABLE and (face_model == "insightface"):
        try:
            print("Initializing InsightFace...")
            init_insightface(with_landmarks=focus_active_speaker)
            insightface_working = True
            print("InsightFace Initialized Successfully.")
        except Exception as e:
            print(f"WARNING: InsightFace Initialization Failed ({e}). Will try MediaPipe.")
            insightface_working = False

    mediapipe_working = False
    use_haar = False
    
    # If insightface failed OR user chose mediapipe, init mediapipe
    should_use_mediapipe = (face_model == "mediapipe") or (face_model == "insightface" and not insightface_working)
    
    if should_use_mediapipe:
        try:
            # Check if solutions is available (it might not be if import failed silently or partial)
            if not hasattr(mp, 'solutions'):
                raise ImportError("mediapipe.solutions not found")
                
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            
            # Try to init with model_selection=0 (Short Range) as a smoketest
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
                pass
            mediapipe_working = True
            print("MediaPipe Initialized Successfully.")
        except Exception as e:
            print(f"WARNING: MediaPipe Initialization Failed ({e}). Switching to OpenCV Haar Cascade.")
            mediapipe_working = False
            use_haar = True
    
    # Logic for MediaPipe replaced by dynamic pass
    # mp_num_faces = 2 if face_mode == "2" else 1  

    import glob
    found_files = sorted(glob.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

    if not found_files:
        print(f"No files found in {cuts_folder}.")
        # Try finding lookahead in case listdir failed? No, glob is fine.
        return

    for input_file in found_files:
        input_filename = os.path.basename(input_file)
        
        # Extract Index
        index = 0
        try:
             parts = input_filename.split('_')
             if parts[0].isdigit(): index = int(parts[0])
             elif input_filename.startswith("output"): # output000
                 idx_str = input_filename[6:9]
                 if idx_str.isdigit(): index = int(idx_str)
        except: pass
        
        output_file = os.path.join(final_folder, f"temp_video_no_audio_{index}.mp4")

        # Determine Final Name (Title)
        base_name_final = input_filename.replace("_original_scale.mp4", "")
        # If legacy name, try to improve it
        if input_filename.startswith("output") and segments_data and index < len(segments_data):
             title = segments_data[index].get("title", f"Segment_{index}")
             safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip().replace(" ", "_")[:60]
             base_name_final = f"{index:03d}_{safe_title}"

        if os.path.exists(input_file):
            success = False
            detected_mode = "1" # Default if detection fails or fallback

            # 0. Layout fit-blur: video inteiro + fundo desfocado, sem deteccao de rosto
            if frame_layout == "fit-blur":
                generate_short_fallback(input_file, output_file, index, project_folder, final_folder,
                                        no_face_mode=no_face_mode, frame_layout="fit-blur")
                success = True

            # 1. Try InsightFace
            if insightface_working and not success:
                try:
                    # Capture returned mode
                    res = generate_short_insightface(input_file, output_file, index, project_folder, final_folder, face_mode=face_mode, detection_period=detection_period, 
                                                     filter_threshold=filter_threshold, two_face_threshold=two_face_threshold, confidence_threshold=confidence_threshold, dead_zone=dead_zone, focus_active_speaker=focus_active_speaker,
                                                     active_speaker_mar=active_speaker_mar, active_speaker_score_diff=active_speaker_score_diff, include_motion=include_motion,
                                                     active_speaker_motion_deadzone=active_speaker_motion_deadzone,
                                                     active_speaker_motion_sensitivity=active_speaker_motion_sensitivity,
                                                     active_speaker_decay=active_speaker_decay,
                                                     no_face_mode=no_face_mode, frame_layout=frame_layout, camera_style=camera_style)
                    if res: detected_mode = res
                    if frame_layout == "fit-faces":
                        detected_mode = "1"  # legenda na posicao padrao
                    success = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"InsightFace processing failed for {input_filename}: {e}")
                    print("Falling back to MediaPipe/Haar...")
            
            # 1b. fit-faces depende do InsightFace; sem ele, mantem o video inteiro com fundo desfocado
            if not success and frame_layout == "fit-faces":
                print("fit-faces requires InsightFace. Falling back to fit-blur for this segment.")
                generate_short_fallback(input_file, output_file, index, project_folder, final_folder,
                                        no_face_mode=no_face_mode, frame_layout="fit-blur")
                success = True

            # 2. Try MediaPipe if InsightFace failed or not available
            if not success and mediapipe_working:
                try:
                    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2) as face_detection, \
                         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh, \
                         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                        
                        generate_short_mediapipe(input_file, output_file, index, face_mode, project_folder, final_folder, face_detection, face_mesh, pose, detection_period=detection_period, no_face_mode=no_face_mode)
                        # We don't easily know detected mode here without return, assuming '1' or '2' based on last frame? 
                        # Ideally function should return as well.
                        detected_mode = "1" # Placeholder, user didn't complain about stats.
                        # detected_mode = str(mp_num_faces) # Error fix: mp_num_faces not defined
                        if face_mode == "2":
                            detected_mode = "2"
                    success = True
                except Exception as e:
                     print(f"MediaPipe processing failed (fallback): {e}")
            
            # 3. Try Haar if others failed
            if not success and (use_haar or (not mediapipe_working and not insightface_working)):
                 try:
                    print("Attempts with Haar Cascade...")
                    generate_short_haar(input_file, output_file, index, project_folder, final_folder, detection_period=detection_period, no_face_mode=no_face_mode)
                    success = True
                 except Exception as e2:
                    print(f"Haar fallback also failed: {e2}")

            # 4. Last Resort: Center Crop
            if not success:
                generate_short_fallback(input_file, output_file, index, project_folder, final_folder, no_face_mode=no_face_mode)
                detected_mode = "1"
                success = True
            
            # Save mode
            face_modes_log[f"output{str(index).zfill(3)}"] = detected_mode

        if success:
             try:
                 new_mp4_name = f"{base_name_final}.mp4"
                 new_mp4_path = os.path.join(final_folder, new_mp4_name)
                 
                 # Source is what finalize_video created
                 # finalize_video creates `final-output{index}_processed.mp4`
                 generated_mp4_name = f"final-output{str(index).zfill(3)}_processed.mp4"
                 generated_mp4_path = os.path.join(final_folder, generated_mp4_name)
                 
                 # 1. Rename MP4
                 if os.path.exists(generated_mp4_path):
                     if os.path.exists(new_mp4_path): os.remove(new_mp4_path)
                     os.rename(generated_mp4_path, new_mp4_path)
                     print(f"Renamed Output to Title: {new_mp4_name}")
                     
                     # 2. Rename JSON Subtitle (if exists and hasn't been renamed by cut_segments)
                     subs_folder = os.path.join(project_folder, "subs")
                     
                     # Check if legacy name exists
                     old_json_name = f"final-output{str(index).zfill(3)}_processed.json"
                     old_json_path = os.path.join(subs_folder, old_json_name)
                     
                     new_json_name = f"{base_name_final}_processed.json"
                     new_json_path = os.path.join(subs_folder, new_json_name)
                     
                     if os.path.exists(old_json_path):
                         if os.path.exists(new_json_path): os.remove(new_json_path)
                         os.rename(old_json_path, new_json_path)
                         print(f"Renamed Subtitles to Title: {new_json_name}")
                         
                     # 3. Rename Timeline JSON
                     # Timeline is temp_video_no_audio_{index}_timeline.json (created by generate_short...)
                     old_timeline_name = f"temp_video_no_audio_{index}_timeline.json"
                     old_timeline_path = os.path.join(final_folder, old_timeline_name)
                     
                     new_timeline_name = f"{base_name_final}_timeline.json"
                     new_timeline_path = os.path.join(final_folder, new_timeline_name)
                     
                     if os.path.exists(old_timeline_path):
                         if os.path.exists(new_timeline_path): os.remove(new_timeline_path)
                         os.rename(old_timeline_path, new_timeline_path)
                         print(f"Renamed Timeline to Title: {new_timeline_name}")
                         
                     # 4. Rename Coords JSON
                     old_coords_name = f"temp_video_no_audio_{index}_coords.json"
                     old_coords_path = os.path.join(final_folder, old_coords_name)
                     
                     new_coords_name = f"{base_name_final}_coords.json"
                     new_coords_path = os.path.join(final_folder, new_coords_name)
                     
                     if os.path.exists(old_coords_path):
                         if os.path.exists(new_coords_path): os.remove(new_coords_path)
                         os.rename(old_coords_path, new_coords_path)
                         print(f"Renamed Coords to Title: {new_coords_name}")
                         
             except Exception as e:
                 print(f"Warning: Could not rename file with title: {e}") 
        
    # Save Face Modes to JSON for subtitle usage
    modes_file = os.path.join(project_folder, "face_modes.json")
    try:
        import json
        with open(modes_file, "w") as f:
            json.dump(face_modes_log, f)
        print(f"Detect Stats saved: {modes_file}")
    except Exception as e:
        print(f"Error saving face modes: {e}")

    save_clip_timings(project_folder, CLIP_TIMINGS)

if __name__ == "__main__":
    edit()