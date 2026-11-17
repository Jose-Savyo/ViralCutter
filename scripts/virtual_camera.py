"""Camera virtual do enquadramento (v2): movimentos de operador, nao de robo.

Regras (inspiradas em como um editor/operador trabalha):
- Zona de conforto: a camera fica parada enquanto o rosto anda pouco (cabeca balancando,
  ruido da deteccao). So reenquadra quando ele sai da zona.
- Confirmacao de salto: uma posicao fora da zona so e aceita depois de se repetir por
  algumas deteccoes (um falso positivo some na deteccao seguinte e nunca move o quadro).
- Movimento com mola criticamente amortecida (SmoothDamp): acelera e freia suave, sem
  passar do alvo, em vez de uma rampa de duracao fixa.
- Troca de pessoa e corte seco, nao panoramica: um salto confirmado maior que ~45% da
  largura do recorte e outra pessoa, e o editor corta. Entre dois cortes ha um tempo minimo.
- Troca de layout (1 <-> 2 rostos) so depois de confirmada e respeitando um tempo minimo
  em cada layout; na troca de cena ela e imediata.

A formula da mola e a SmoothDamp classica (ver Game Programming Gems 4, cap. 1.10);
a implementacao abaixo e propria.
"""
import math

import numpy as np

# zone: raio da zona de conforto (fracao da altura do quadro); smooth_time: tempo da mola (s);
# confirm_s: tempo minimo de confirmacao de um salto; min_dwell_s: tempo minimo entre cortes/trocas.
CAMERA_STYLES = {
    "calm":    {"zone": 0.13, "smooth_time": 0.50, "confirm_s": 0.50, "min_dwell_s": 2.0},
    "normal":  {"zone": 0.11, "smooth_time": 0.35, "confirm_s": 0.35, "min_dwell_s": 1.5},
    "dynamic": {"zone": 0.08, "smooth_time": 0.25, "confirm_s": 0.20, "min_dwell_s": 1.0},
}
CUT_JUMP_RATIO = 0.45      # salto confirmado > 45% da largura do recorte = outra pessoa -> corte seco
LAYOUT_CONFIRM_S = 0.5     # nova quantidade de rostos precisa persistir esse tempo
STALE_SLOT_S = 1.5         # rosto que sumiu por mais que isso libera a troca de layout
MAX_SPEED_RATIO = 0.6      # velocidade maxima da panoramica: 60% da largura do quadro por segundo
SETTLE_PX = 3.0            # a menos de 3 px do alvo e devagar: considera parado
SETTLE_SPEED = 20.0        # px/s


def crop_size(frame_w, frame_h, aspect=9.0 / 16.0):
    """Maior recorte 9:16 que cabe no quadro."""
    ch = float(frame_h)
    cw = ch * aspect
    if cw > frame_w:
        cw = float(frame_w)
        ch = cw / aspect
    return cw, ch


def crop_window(frame_w, frame_h, cx, cy, zoom=1.0, headroom=0.42):
    """Janela (x0, y0, w, h) em ponto flutuante centrada no rosto. zoom > 1 aproxima (punch-in)."""
    cw, ch = crop_size(frame_w, frame_h)
    zoom = max(1.0, float(zoom))
    cw, ch = cw / zoom, ch / zoom
    x0 = min(max(cx - cw / 2.0, 0.0), frame_w - cw)
    y0 = min(max(cy - headroom * ch, 0.0), frame_h - ch)
    return x0, y0, cw, ch


def smooth_damp(current, target, velocity, smooth_time, max_speed, dt):
    """Mola criticamente amortecida, vetorizada (numpy). Retorna (posicao, velocidade)."""
    smooth_time = max(1e-4, float(smooth_time))
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    max_change = max_speed * smooth_time
    change = np.clip(change, -max_change, max_change)
    goal = current - change
    temp = (velocity + omega * change) * dt
    new_vel = (velocity - omega * temp) * exp
    out = goal + (change + temp) * exp
    overshoot = (target - current > 0) == (out > target)
    out = np.where(overshoot, target, out)
    new_vel = np.where(overshoot, 0.0, new_vel)
    return out, new_vel


def _to_cwh(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)])


def _to_bbox(c):
    cx, cy, w, h = c
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _dist(a, b):
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


class VirtualCamera:
    """Uma camera (um rosto enquadrado)."""

    def __init__(self, zone, smooth_time, confirm_s, max_speed):
        self.zone = zone
        self.smooth_time = smooth_time
        self.confirm_s = confirm_s
        self.max_speed = max_speed
        self.pos = None
        self.vel = np.zeros(4)
        self.anchor = None
        self.pending = None
        self.pending_since = 0.0
        self.pending_count = 0

    def snap(self, bbox):
        c = _to_cwh(bbox)
        self.pos = c.copy()
        self.anchor = c.copy()
        self.vel = np.zeros(4)
        self.pending = None
        self.pending_count = 0

    def observe(self, bbox, t):
        """Nova deteccao. Retorna None (dentro da zona), 'pending' ou 'jump' (salto confirmado)."""
        c = _to_cwh(bbox)
        if self.pos is None:
            self.snap(bbox)
            return None
        # Tamanho acompanha devagar, sem "respirar" com o ruido da deteccao
        if abs(c[3] - self.anchor[3]) / max(1.0, self.anchor[3]) > 0.12:
            self.anchor[2:] = 0.6 * self.anchor[2:] + 0.4 * c[2:]
        if _dist(c, self.anchor) <= self.zone:
            self.pending = None
            self.pending_count = 0
            return None
        if self.pending is not None and _dist(c, self.pending) <= self.zone:
            self.pending_count += 1
            self.pending = c
        else:
            self.pending = c
            self.pending_count = 1
            self.pending_since = t
        if self.pending_count >= 2 and (t - self.pending_since) >= self.confirm_s - 1e-6:
            return "jump"
        return "pending"

    def pending_distance(self):
        return _dist(self.pending, self.anchor) if self.pending is not None else 0.0

    def accept_pan(self):
        self.anchor = self.pending.copy()
        self.pending = None
        self.pending_count = 0

    def accept_cut(self):
        self.snap(_to_bbox(self.pending))

    def step(self, dt):
        self.pos, self.vel = smooth_damp(self.pos, self.anchor, self.vel, self.smooth_time, self.max_speed, dt)
        # Chegou: para de vez (sem ficar "flutuando" frações de pixel no fim da mola)
        if np.all(np.abs(self.pos - self.anchor) < SETTLE_PX) and np.all(np.abs(self.vel) < SETTLE_SPEED):
            self.pos = self.anchor.copy()
            self.vel = np.zeros(4)
        return _to_bbox(self.pos)


class CameraRig:
    """Conjunto de cameras (1 ou 2 rostos) + regras de troca de layout e de corte."""

    def __init__(self, frame_w, frame_h, fps, style="normal", dead_zone=0.0):
        p = CAMERA_STYLES.get(style or "normal", CAMERA_STYLES["normal"])
        self.style = style if style in CAMERA_STYLES else "normal"
        self.frame_w, self.frame_h, self.fps = frame_w, frame_h, fps
        self.crop_w, self.crop_h = crop_size(frame_w, frame_h)
        self.zone = max(float(dead_zone or 0.0), p["zone"] * frame_h)
        self.smooth_time = p["smooth_time"]
        self.confirm_s = p["confirm_s"]
        self.min_dwell = p["min_dwell_s"]
        self.cut_px = CUT_JUMP_RATIO * self.crop_w
        self.max_speed = MAX_SPEED_RATIO * frame_w
        self.cams = []
        self.count = None
        self.last_seen = []
        self.last_cut_t = -1e9
        self.last_layout_t = -1e9
        self.layout_candidate = None
        self.layout_candidate_since = 0.0
        self.stats = {"cortes_pessoa": 0, "panoramicas": 0, "trocas_layout": 0}

    def reset(self, t=None):
        """Troca de cena (ou rosto perdido por muito tempo): a proxima deteccao entra com corte seco."""
        self.cams = []
        self.count = None
        self.last_seen = []
        self.layout_candidate = None
        if t is not None:
            self.last_cut_t = t
            self.last_layout_t = t

    def _new_cam(self):
        return VirtualCamera(self.zone, self.smooth_time, self.confirm_s, self.max_speed)

    def _set_layout(self, detections, t):
        if self.count is not None and self.count != len(detections):
            self.stats["trocas_layout"] += 1
        self.cams = [self._new_cam() for _ in detections]
        for cam, b in zip(self.cams, detections):
            cam.snap(b)
        self.count = len(detections)
        self.last_seen = [t] * self.count
        self.last_layout_t = t
        self.layout_candidate = None

    def _map_to_slots(self, detections):
        """Layout mantido: associa cada slot a deteccao mais proxima da sua ancora."""
        mapped, used = {}, set()
        for slot, cam in enumerate(self.cams):
            best, best_d = None, None
            for i, b in enumerate(detections):
                if i in used:
                    continue
                d = _dist(_to_cwh(b), cam.anchor)
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best is not None:
                mapped[slot] = detections[best]
                used.add(best)
        return mapped

    def observe(self, detections, t):
        n = len(detections)
        if n == 0:
            return
        if self.count is None:
            self._set_layout(detections, t)
            return
        if n != self.count:
            if self.layout_candidate != n:
                self.layout_candidate = n
                self.layout_candidate_since = t
            mapped = self._map_to_slots(detections)
            unmatched = [s for s in range(self.count) if s not in mapped]
            stale = any(t - self.last_seen[s] > STALE_SLOT_S for s in unmatched)
            confirmed = (t - self.layout_candidate_since) >= LAYOUT_CONFIRM_S
            dwell_ok = (t - self.last_layout_t) >= self.min_dwell
            if (confirmed and dwell_ok) or stale:
                self._set_layout(detections, t)
                return
        else:
            self.layout_candidate = None
            mapped = dict(enumerate(detections))

        for slot, bbox in mapped.items():
            self.last_seen[slot] = t
            cam = self.cams[slot]
            if cam.observe(bbox, t) != "jump":
                continue
            if cam.pending_distance() >= self.cut_px:
                if t - self.last_cut_t >= self.min_dwell:
                    cam.accept_cut()
                    self.last_cut_t = t
                    self.stats["cortes_pessoa"] += 1
                # senao: segura o plano ate cumprir o tempo minimo (o salto continua pendente)
            else:
                cam.accept_pan()
                self.stats["panoramicas"] += 1

    def step(self, dt):
        return [cam.step(dt) for cam in self.cams]
