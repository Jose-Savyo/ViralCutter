"""Punch-in: um zoom curto de enfase nos picos da fala (layout crop, um rosto).

Um recorte parado por 45 s e o que mais denuncia "video nao editado". O zoom e pequeno
(12%: aproxima sem cortar a cabeca) e assimetrico, como um operador faz: entra rapido,
segura, sai devagar. Parametros calibrados no openshorts (punch_in.py, MIT) para dar
cerca de 2 a 4 zooms por minuto; implementacao propria.
"""
import subprocess

import numpy as np

MAX_ZOOM = 1.12
RISE_SECONDS = 0.25
HOLD_SECONDS = 1.30
FALL_SECONDS = 0.55
MIN_GAP_SECONDS = 18.0     # nunca dois zooms em menos de 18 s
BEAT_WINDOW = 0.2          # resolucao do envelope de audio
BEAT_PROMINENCE = 0.60     # pico precisa passar a mediana em 60% da distancia ate o maximo
EDGE_SECONDS = 1.5         # nada de zoom colado no inicio/fim do corte


def emphasis_times(media_path, duration, sr=8000):
    """Instantes (s) dos picos de fala do corte, a partir do envelope de volume."""
    cmd = ["ffmpeg", "-v", "error", "-i", media_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception:
        return []
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = int(BEAT_WINDOW * sr)
    if audio.size < win * 10:
        return []
    n = audio.size // win
    env = np.sqrt((audio[:n * win].reshape(n, win) ** 2).mean(axis=1))
    med, peak = float(np.median(env)), float(env.max())
    if peak <= med:
        return []
    threshold = med + BEAT_PROMINENCE * (peak - med)
    candidates = [i for i in range(1, n - 1) if env[i] >= threshold and env[i] >= env[i - 1] and env[i] >= env[i + 1]]
    candidates.sort(key=lambda i: env[i], reverse=True)   # os picos mais fortes primeiro
    chosen = []
    total = min(float(duration), n * BEAT_WINDOW)
    for i in candidates:
        t = i * BEAT_WINDOW
        if t < EDGE_SECONDS or t > total - EDGE_SECONDS - HOLD_SECONDS:
            continue
        if all(abs(t - c) >= MIN_GAP_SECONDS for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def zoom_curve(n_frames, fps, times, max_zoom=MAX_ZOOM):
    """Zoom por quadro (1.0 = sem zoom). Entra rapido, segura, sai devagar."""
    z = np.ones(int(n_frames), dtype=np.float32)
    if not times or fps <= 0:
        return z
    t = np.arange(int(n_frames)) / float(fps)
    for start in times:
        rise = _ease((t - start) / RISE_SECONDS)
        fall = 1.0 - _ease((t - (start + RISE_SECONDS + HOLD_SECONDS)) / FALL_SECONDS)
        shape = np.where(t < start + RISE_SECONDS, rise, fall)
        shape = np.where((t < start) | (t > start + RISE_SECONDS + HOLD_SECONDS + FALL_SECONDS), 0.0, shape)
        z = np.maximum(z, 1.0 + (max_zoom - 1.0) * shape)
    return z
