"""Metricas para comparar versoes do ViralCutter (antes/depois).

Uso:
  python tools/metricas.py camera  <arquivo_coords.json> [--fps 29.97] [--json]
  python tools/metricas.py limites <pasta_do_projeto> [--json]
  python tools/metricas.py tempos  <pasta_do_projeto>

camera:  movimentos de camera por minuto, reversoes por segundo, deslocamento em px/s,
         cortes secos e trocas de layout, a partir do <corte>_coords.json do enquadramento.
limites: cortes que comecam/terminam no meio de uma palavra e que terminam fora do fim da frase,
         a partir de viral_segments.txt + input.json (palavras com tempo).
tempos:  resumo de timings.json e timings_clips.json.
"""
import argparse
import json
import os
import re
import sys

SENTENCE_END = re.compile(r'[.!?…]["\'”»)\]]*$')


def _crop_center(entry, crop_w):
    if entry.get("crop"):
        x, _y, w, _h = entry["crop"]
        return x + w / 2.0
    faces = entry.get("faces") or []
    if not faces:
        return None
    src_w = (entry.get("src_size") or [1920, 1080])[0]
    f = faces[0]
    cx = (float(f[0]) + float(f[2])) / 2.0
    return min(max(cx, crop_w / 2.0), src_w - crop_w / 2.0)


def scene_cut_frames(video_path):
    """Quadros com troca de cena, pelo mesmo criterio do enquadramento (histograma HSV)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import cv2
    from scripts.edit_video import frame_histogram, is_scene_cut
    cap = cv2.VideoCapture(video_path)
    cuts, prev, i = set(), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = frame_histogram(frame)
        if is_scene_cut(prev, cur):
            cuts.add(i)
        prev, i = cur, i + 1
    cap.release()
    return cuts


def camera_metrics(coords, fps=29.97, scene_cuts=None):
    if not coords:
        return {}
    src_w, src_h = (coords[0].get("src_size") or [1920, 1080])[:2]
    crop_w = src_h * 9.0 / 16.0
    cut_jump = 0.25 * crop_w          # salto maior que isso num quadro = corte seco, nao movimento
    xs = [_crop_center(e, crop_w) for e in coords]
    layouts = [e.get("layout") or (f"{len(e.get('faces') or [])}f") for e in coords]

    moving_frames = cuts = movements = reversals = scene_jumps = 0
    travel = 0.0
    prev_dir = 0
    in_move = False
    still = 0
    end_after = max(1, int(round(0.25 * fps)))   # movimento so termina depois de 0,25 s parado
    for i in range(1, len(xs)):
        a, b = xs[i - 1], xs[i]
        if a is None or b is None:
            in_move = False
            continue
        dx = b - a
        if scene_cuts is not None and i in scene_cuts:
            if abs(dx) > 0.5:
                scene_jumps += 1
            in_move = False
            prev_dir = 0
            continue
        if abs(dx) >= cut_jump:
            cuts += 1
            in_move = False
            prev_dir = 0
            continue
        if abs(dx) > 0.5:
            moving_frames += 1
            travel += abs(dx)
            still = 0
            d = 1 if dx > 0 else -1
            if not in_move:
                movements += 1
                in_move = True
            if prev_dir and d != prev_dir:
                reversals += 1
            prev_dir = d
        else:
            still += 1
            if still >= end_after:
                in_move = False
    layout_changes = sum(1 for i in range(1, len(layouts)) if layouts[i] != layouts[i - 1])
    duration = len(xs) / float(fps)
    minutes = duration / 60.0 if duration else 1.0
    return {
        "duracao_s": round(duration, 2),
        "movimentos_por_min": round(movements / minutes, 2),
        "reversoes_por_s": round(reversals / duration, 3) if duration else 0.0,
        "deslocamento_px_s": round(travel / duration, 1) if duration else 0.0,
        "quadros_em_movimento_pct": round(100.0 * moving_frames / max(1, len(xs) - 1), 1),
        "cortes_secos": cuts,
        "saltos_em_troca_de_cena": scene_jumps,
        "trocas_de_layout_por_min": round(layout_changes / minutes, 2),
        "quadros_sem_rosto_pct": round(100.0 * sum(1 for x in xs if x is None) / len(xs), 1),
    }


def _load_words(project):
    path = os.path.join(project, "input.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []) or []:
            if "start" in w and "end" in w:
                words.append((float(w["start"]), float(w["end"]), str(w.get("word", "")).strip()))
    words.sort()
    return words


def boundary_metrics(project):
    with open(os.path.join(project, "viral_segments.txt"), encoding="utf-8") as f:
        segs = json.load(f).get("segments", [])
    words = _load_words(project)
    rows = []
    for s in segs:
        st, en = float(s["start_time"]), float(s["end_time"])
        start_mid = any(w0 < st - 0.02 and w1 > st + 0.02 for w0, w1, _ in words)
        end_mid = any(w0 < en - 0.02 and w1 > en + 0.02 for w0, w1, _ in words)
        inside = [w for w in words if w[0] >= st - 0.02 and w[1] <= en + 0.02]
        last = inside[-1][2] if inside else ""
        first_gap = (inside[0][0] - st) if inside else None
        last_gap = (en - inside[-1][1]) if inside else None
        rows.append({
            "titulo": s.get("title", "")[:50],
            "inicio_no_meio_da_palavra": start_mid,
            "fim_no_meio_da_palavra": end_mid,
            "fim_de_frase": bool(SENTENCE_END.search(last)),
            "respiro_inicio_s": round(first_gap, 2) if first_gap is not None else None,
            "respiro_fim_s": round(last_gap, 2) if last_gap is not None else None,
            "hook_igual_titulo": (s.get("hook") or "").strip() == (s.get("title") or "").strip(),
        })
    n = len(rows) or 1
    summary = {
        "cortes": len(rows),
        "inicio_no_meio_da_palavra": sum(r["inicio_no_meio_da_palavra"] for r in rows),
        "fim_no_meio_da_palavra": sum(r["fim_no_meio_da_palavra"] for r in rows),
        "fim_fora_do_fim_da_frase": sum(not r["fim_de_frase"] for r in rows),
        "hook_igual_titulo": sum(r["hook_igual_titulo"] for r in rows),
        "respiro_inicio_medio_s": round(sum((r["respiro_inicio_s"] or 0) for r in rows) / n, 2),
        "respiro_fim_medio_s": round(sum((r["respiro_fim_s"] or 0) for r in rows) / n, 2),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("camera")
    c.add_argument("coords")
    c.add_argument("--fps", type=float, default=29.97)
    c.add_argument("--json", action="store_true")
    c.add_argument("--video", help="corte de origem (_original_scale.mp4): desconta as trocas de cena")
    l = sub.add_parser("limites")
    l.add_argument("projeto")
    l.add_argument("--json", action="store_true")
    t = sub.add_parser("tempos")
    t.add_argument("projeto")
    a = ap.parse_args()

    if a.cmd == "camera":
        with open(a.coords, encoding="utf-8") as f:
            m = camera_metrics(json.load(f), a.fps, scene_cut_frames(a.video) if a.video else None)
        print(json.dumps(m, ensure_ascii=False) if a.json else "\n".join(f"{k}: {v}" for k, v in m.items()))
    elif a.cmd == "limites":
        summary, rows = boundary_metrics(a.projeto)
        if a.json:
            print(json.dumps({"resumo": summary, "cortes": rows}, ensure_ascii=False))
        else:
            for r in rows:
                print(r)
            print("\n".join(f"{k}: {v}" for k, v in summary.items()))
    elif a.cmd == "tempos":
        for name in ("timings.json", "timings_clips.json"):
            path = os.path.join(a.projeto, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    print(name, json.dumps(json.load(f), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
