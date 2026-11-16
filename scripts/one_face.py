import cv2
import numpy as np
import os
import subprocess
import mediapipe as mp

def crop_and_resize_single_face(frame, face):
        frame_height, frame_width = frame.shape[:2]

        x, y, w, h = face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Cálculo da proporção desejada (9:16)
        target_aspect_ratio = 9 / 16

        # Cálculo da área de corte para evitar barras pretas
        if frame_width / frame_height > target_aspect_ratio:
            new_width = int(frame_height * target_aspect_ratio)
            new_height = frame_height
        else:
            new_width = frame_width
            new_height = int(frame_width / target_aspect_ratio)

        # Garantir que o corte esteja dentro dos limites
        crop_x = max(0, min(face_center_x - new_width // 2, frame_width - new_width))
        crop_y = max(0, min(face_center_y - new_height // 2, frame_height - new_height))
        crop_x2 = crop_x + new_width
        crop_y2 = crop_y + new_height

        # Recorte e redimensionamento para 1080x1920 (9:16)
        crop_img = frame[crop_y:crop_y2, crop_x:crop_x2]
        resized = cv2.resize(crop_img, (1080, 1920), interpolation=cv2.INTER_AREA)

        return resized

def resize_with_padding(frame):
        frame_height, frame_width = frame.shape[:2]
        target_aspect_ratio = 9 / 16

        if frame_width / frame_height > target_aspect_ratio:
            new_width = frame_width
            new_height = int(frame_width / target_aspect_ratio)
        else:
            new_height = frame_height
            new_width = int(frame_height * target_aspect_ratio)

        # Criação de uma tela preta
        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Cálculo das margens
        pad_top = (new_height - frame_height) // 2
        pad_left = (new_width - frame_width) // 2

        # Colocar o frame original na tela
        result[pad_top:pad_top+frame_height, pad_left:pad_left+frame_width] = frame

        # Redimensionar para as dimensões finais
        return cv2.resize(result, (1080, 1920), interpolation=cv2.INTER_AREA)

def detect_face_or_body(frame, face_detection, face_mesh, pose):
    # Converter a imagem para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a detecção de rosto
    results_face_detection = face_detection.process(frame_rgb)
    results_face_mesh = face_mesh.process(frame_rgb)
    results_pose = pose.process(frame_rgb)

    detections = []

    # Usar a detecção de rosto se disponível
    if results_face_detection.detections:
        # Usar o primeiro rosto detectado
        detection = results_face_detection.detections[0]
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * frame.shape[1])
        y_min = int(bbox.ymin * frame.shape[0])
        width = int(bbox.width * frame.shape[1])
        height = int(bbox.height * frame.shape[0])
        detections.append((x_min, y_min, width, height))
    
    # Usar landmarks do face mesh se disponível
    if results_face_mesh.multi_face_landmarks:
        landmarks = results_face_mesh.multi_face_landmarks[0].landmark
        # Coordenadas do rosto baseadas nos pontos-chave (landmarks)
        x_coords = [int(landmark.x * frame.shape[1]) for landmark in landmarks]
        y_coords = [int(landmark.y * frame.shape[0]) for landmark in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        detections.append((x_min, y_min, width, height))

    # Se nenhum rosto for detectado, usar a pose para estimar o corpo
    if results_pose.pose_landmarks:
        x_coords = [lmk.x for lmk in results_pose.pose_landmarks.landmark]
        y_coords = [lmk.y for lmk in results_pose.pose_landmarks.landmark]
        x_min = int(min(x_coords) * frame.shape[1])
        x_max = int(max(x_coords) * frame.shape[1])
        y_min = int(min(y_coords) * frame.shape[0])
        y_max = int(max(y_coords) * frame.shape[0])
        width = x_max - x_min
        height = y_max - y_min
        detections.append((x_min, y_min, width, height))

    # Se nada for detectado, retornar uma lista vazia
    return detections if detections else None


def crop_center_zoom(frame):
    """
    Crops the center of the frame to fill 9:16 aspect ratio (Zoom effect).
    """
    frame_height, frame_width = frame.shape[:2]
    target_aspect_ratio = 9 / 16
    
    # Calculate crop dimensions to FILL the target ratio
    if frame_width / frame_height > target_aspect_ratio:
        # Source is wider than target (e.g. 16:9 source, 9:16 target) -> Crop Width
        new_width = int(frame_height * target_aspect_ratio)
        new_height = frame_height
    else:
        # Source is taller than target -> Crop Height
        new_width = frame_width
        new_height = int(frame_width / target_aspect_ratio)
        
    start_x = (frame_width - new_width) // 2
    start_y = (frame_height - new_height) // 2
    
    # Ensure bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    
    crop_img = frame[start_y:start_y+new_height, start_x:start_x+new_width]
    
    # Resize to final 1080x1920
    return cv2.resize(crop_img, (1080, 1920), interpolation=cv2.INTER_AREA)



# ---------------------------------------------------------------------------
# Layouts "video completo" (frame_layout = fit-blur / fit-faces)
# O quadro original inteiro fica no centro do 9:16; as faixas livres em cima e
# embaixo recebem um fundo desfocado (fit-blur) ou zoom nos rostos (fit-faces).
# ---------------------------------------------------------------------------
TARGET_W, TARGET_H = 1080, 1920


def _center_crop_region(frame_w, frame_h, aspect):
    """Maior retangulo centralizado com a proporcao `aspect` (w/h). Retorna (x, y, w, h)."""
    if frame_w / frame_h > aspect:
        h = frame_h
        w = int(round(h * aspect))
    else:
        w = frame_w
        h = int(round(w / aspect))
    return (frame_w - w) // 2, (frame_h - h) // 2, w, h


def blur_background(frame, width=TARGET_W, height=TARGET_H, darken=0.55):
    """Fundo: zoom central do proprio quadro, desfocado e escurecido (feito em baixa resolucao)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = _center_crop_region(fw, fh, width / height)
    small = cv2.resize(frame[y:y + h, x:x + w], (max(1, width // 8), max(1, height // 8)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    bg = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.convertScaleAbs(bg, alpha=darken, beta=0)


def fit_to_width(frame, width=TARGET_W):
    """Redimensiona o quadro inteiro para `width` de largura (altura par)."""
    fh, fw = frame.shape[:2]
    h = int(round(width * fh / fw))
    h -= h % 2
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def crop_band(frame, bbox, band_w, band_h, face_scale=2.4, crop_aspect=1.0):
    """
    Zoom no rosto para uma faixa band_w x band_h.
    Recorta uma regiao quase quadrada (crop_aspect = w/h) em volta do rosto, para nao pegar
    o que esta ao lado (player, outras webcams), e centraliza na faixa sobre um fundo desfocado
    do proprio recorte. bbox = [x1, y1, x2, y2]; face_scale = altura do recorte / altura do rosto.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    face_h = max(1.0, y2 - y1)
    crop_h = min(max(face_h * face_scale, 48.0), float(fh))
    crop_w = min(crop_h * crop_aspect, float(fw))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + 0.15 * face_h  # desce um pouco para mostrar ombros
    cx0 = int(round(min(max(cx - crop_w / 2, 0), fw - crop_w)))
    cy0 = int(round(min(max(cy - crop_h / 2, 0), fh - crop_h)))
    crop = frame[cy0:cy0 + int(crop_h), cx0:cx0 + int(crop_w)]
    if crop.size == 0:
        return blur_background(frame, band_w, band_h)

    ch, cw = crop.shape[:2]
    scale = band_h / ch
    out_w = int(round(cw * scale))
    if out_w >= band_w:  # recorte ja preenche a largura
        out_h = int(round(band_w * ch / cw))
        fg = cv2.resize(crop, (band_w, max(out_h, band_h)), interpolation=cv2.INTER_LINEAR)
        y = (fg.shape[0] - band_h) // 2
        return fg[y:y + band_h]
    fg = cv2.resize(crop, (out_w, band_h), interpolation=cv2.INTER_LINEAR)
    canvas = blur_background(crop, band_w, band_h)
    x = (band_w - out_w) // 2
    canvas[:, x:x + out_w] = fg
    return canvas


def compose_fit_blur(frame):
    """Video inteiro no centro, fundo desfocado atras."""
    fg = fit_to_width(frame)
    if fg.shape[0] >= TARGET_H:  # fonte ja vertical: nada a encaixar
        return crop_center_zoom(frame)
    canvas = blur_background(frame)
    y = (TARGET_H - fg.shape[0]) // 2
    canvas[y:y + fg.shape[0]] = fg
    return canvas


def compose_fit_faces(frame, faces):
    """
    Video inteiro no centro; faixa de cima = zoom no rosto 1, faixa de baixo = zoom no rosto 2.
    faces: lista de bboxes [x1, y1, x2, y2] (0, 1 ou 2+). Sem rosto para a faixa -> fundo desfocado.
    """
    fg = fit_to_width(frame)
    if fg.shape[0] >= TARGET_H:
        return crop_center_zoom(frame)
    free = TARGET_H - fg.shape[0]
    top_h = free // 2
    bottom_h = free - top_h
    faces = list(faces or [])

    canvas = None
    if len(faces) < 2:
        canvas = blur_background(frame)
    else:
        canvas = np.empty((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    if top_h > 0 and len(faces) >= 1:
        canvas[:top_h] = crop_band(frame, faces[0], TARGET_W, top_h)
    if bottom_h > 0 and len(faces) >= 2:
        canvas[TARGET_H - bottom_h:] = crop_band(frame, faces[1], TARGET_W, bottom_h)
    canvas[top_h:top_h + fg.shape[0]] = fg
    return canvas
