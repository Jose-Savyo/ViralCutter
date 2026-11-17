"""Compressao de pausas e remocao de vicios ("é...", "hum") dentro de um corte.

Uma pausa so e encurtada quando e segura: o trecho sem fala tambem precisa estar silencioso
(sem risada, musica ou efeito) e visualmente parado (em gameplay, esporte ou culinaria o
trecho sem fala costuma ser a acao). As duas checagens usam audio mono a 8 kHz e quadros
64x36 em cinza, entao sao baratas.

A remocao acontece no mesmo encode do corte (filtros select/aselect do ffmpeg) e os tempos
das palavras da legenda sao remapeados para a nova linha do tempo.
"""
import re
import subprocess

import numpy as np

PAUSE_MIN_SECONDS = 0.7      # pausa menor que isso fica como esta
PAUSE_KEEP_SECONDS = 0.25    # quanto da pausa fica (respiro natural)
QUIET_RATIO = 0.35           # pausa precisa ter RMS < 35% do nivel da fala
STILL_DIFF = 6.0             # diferenca media entre quadros 64x36 (0-255) abaixo disso = parado
FILLER_MARGIN = 0.03
# So hesitacoes que nao sao palavras do portugues. "é" (verbo), "um" (artigo), "ah", "tipo" e "né"
# tambem sao palavras de verdade: so saem quando isoladas por pausas dos dois lados.
FILLERS_ALWAYS = {"éé", "ééé", "eh", "ehh", "ahn", "hã", "hum", "hmm", "hm", "uh", "uhm", "erm"}
FILLERS_ISOLATED = {"é", "um", "ah", "ã", "er", "tipo", "né"}
ISOLATED_GAP = 0.15


def _norm(word):
    return re.sub(r"[^\wáàâãéêíóôõúç]", "", str(word).lower())


def _audio_window(input_file, start, duration, sr=8000):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", input_file,
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception:
        return None, sr
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def _rms(samples):
    return float(np.sqrt(np.mean(samples * samples))) if samples is not None and samples.size else 0.0


def _visual_still(input_file, start, duration):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{max(0.3, duration):.3f}", "-i", input_file,
           "-vf", "fps=4,scale=64:36,format=gray", "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception:
        return False
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (64 * 36)
    if n < 2:
        return True
    frames = frames[:n * 64 * 36].reshape(n, 36, 64).astype(np.float32)
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return float(np.median(diffs)) < STILL_DIFF


def plan_removals(words, clip_start, clip_end, input_file, remove_pauses=True, remove_fillers=False):
    """
    words: [(inicio, fim, texto)] em tempo do video original.
    Retorna intervalos a remover, relativos ao inicio do corte: [(a, b), ...] ordenados e sem sobreposicao.
    """
    ws = [w for w in words if w[1] > clip_start and w[0] < clip_end]
    if len(ws) < 2:
        return []
    removals = []
    audio, sr = (None, 8000)
    if remove_pauses:
        audio, sr = _audio_window(input_file, clip_start, clip_end - clip_start)
        speech = [audio[int((w0 - clip_start) * sr):int((w1 - clip_start) * sr)]
                  for w0, w1, _ in ws if audio is not None]
        speech_level = float(np.median([_rms(s) for s in speech if s.size])) if speech else 0.0
        for (a0, a1, _), (b0, b1, _) in zip(ws, ws[1:]):
            gap = b0 - a1
            if gap < PAUSE_MIN_SECONDS:
                continue
            cut_a = a1 + PAUSE_KEEP_SECONDS / 2
            cut_b = b0 - PAUSE_KEEP_SECONDS / 2
            if audio is not None and speech_level > 0:
                seg = audio[int((a1 - clip_start) * sr):int((b0 - clip_start) * sr)]
                if _rms(seg) > QUIET_RATIO * speech_level:
                    continue  # tem som na pausa (risada, musica, efeito): fica
            if not _visual_still(input_file, a1, gap):
                continue  # tem acao na tela: fica
            removals.append((cut_a - clip_start, cut_b - clip_start))
    if remove_fillers:
        for i, (w0, w1, txt) in enumerate(ws):
            n = _norm(txt)
            if not n:
                continue
            gap_before = w0 - ws[i - 1][1] if i > 0 else 1.0
            gap_after = ws[i + 1][0] - w1 if i + 1 < len(ws) else 1.0
            isolated = gap_before >= ISOLATED_GAP and gap_after >= ISOLATED_GAP
            if n in FILLERS_ALWAYS or (n in FILLERS_ISOLATED and isolated):
                # Remove a palavra e encurta as pausas em volta (senao as duas viram uma pausa longa)
                a = w0 - FILLER_MARGIN
                b = w1 + FILLER_MARGIN
                if i > 0 and gap_before > PAUSE_KEEP_SECONDS / 2:
                    a = min(a, ws[i - 1][1] + PAUSE_KEEP_SECONDS / 2)
                if i + 1 < len(ws) and gap_after > PAUSE_KEEP_SECONDS / 2:
                    b = max(b, ws[i + 1][0] - PAUSE_KEEP_SECONDS / 2)
                removals.append((max(0.0, a - clip_start), b - clip_start))
    removals.sort()
    merged = []
    for a, b in removals:
        if b - a < 0.05:
            continue
        if merged and a <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def keep_intervals(removals, duration):
    """Complemento das remocoes dentro de [0, duracao]."""
    keep, t = [], 0.0
    for a, b in removals:
        if a > t:
            keep.append((t, a))
        t = max(t, b)
    if t < duration:
        keep.append((t, duration))
    return keep


def select_filters(keep):
    """Filtros de video/audio do ffmpeg que mantem so os intervalos `keep` e recompoem os tempos."""
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{expr}',asetpts=N/SR/TB"
    return vf, af


def remap_time(t, removals):
    """Tempo no corte original -> tempo no corte sem as remocoes. None se cair dentro de uma remocao."""
    shift = 0.0
    for a, b in removals:
        if t >= b:
            shift += b - a
        elif t > a:
            return None
        else:
            break
    return t - shift


def remap_subtitle_json(data, removals):
    """Ajusta os tempos (relativos ao corte) das palavras/segmentos e descarta as palavras removidas."""
    if not removals:
        return data
    new_segments = []
    for seg in data.get("segments", []):
        words = []
        for w in seg.get("words", []) or []:
            if "start" in w and "end" in w:
                s = remap_time(float(w["start"]), removals)
                e = remap_time(float(w["end"]), removals)
                if s is None:
                    continue  # palavra removida (vicio)
                if e is None:
                    e = s + max(0.05, float(w["end"]) - float(w["start"]))
                words.append({**w, "start": round(s, 3), "end": round(max(e, s + 0.02), 3)})
            else:
                words.append(w)
        timed = [w for w in words if "start" in w]
        if not timed and seg.get("words"):
            continue
        new_seg = dict(seg)
        new_seg["words"] = words
        if timed:
            new_seg["start"] = timed[0]["start"]
            new_seg["end"] = timed[-1]["end"]
            new_seg["text"] = " ".join(str(w.get("word", "")).strip() for w in words).strip()
        new_segments.append(new_seg)
    return {**data, "segments": new_segments}


def load_words(project_folder):
    """Palavras com tempo do input.json: [(inicio, fim, texto)]."""
    import json
    import os
    path = os.path.join(project_folder, "input.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    words = [(float(w["start"]), float(w["end"]), str(w.get("word", "")))
             for seg in data.get("segments", []) for w in seg.get("words", []) or []
             if w.get("start") is not None and w.get("end") is not None]
    return sorted(words)
