"""Medicao de tempo do pipeline: por etapa (main) e por corte (enquadramento).

Mensagens com o prefixo [TEMPO] para facilitar a leitura no log do Colab.
Os totais sao salvos em timings.json (etapas) e timings_clips.json (cortes) na pasta do projeto.
"""
import json
import os
import time
from contextlib import contextmanager


def fmt_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


class StageTimer:
    """Cronometra as etapas do pipeline em sequencia: start() de uma etapa encerra a anterior."""

    def __init__(self):
        self.t_start = time.perf_counter()
        self.stages = []
        self._current = None
        self._t0 = None

    def start(self, name):
        self.stop()
        self._current = name
        self._t0 = time.perf_counter()
        print(f"[TEMPO] Etapa '{name}' iniciada", flush=True)

    def stop(self):
        if self._current is None:
            return
        elapsed = time.perf_counter() - self._t0
        self.stages.append((self._current, elapsed))
        print(f"[TEMPO] Etapa '{self._current}' concluida em {fmt_seconds(elapsed)}", flush=True)
        self._current = None

    def report(self, project_folder=None):
        self.stop()
        total = time.perf_counter() - self.t_start
        if not self.stages:
            return
        width = max(len(name) for name, _ in self.stages)
        print("[TEMPO] ===== Resumo por etapa =====", flush=True)
        for name, elapsed in self.stages:
            share = 100.0 * elapsed / total if total else 0.0
            print(f"[TEMPO] {name.ljust(width)}  {fmt_seconds(elapsed).rjust(9)}  ({share:4.1f}%)", flush=True)
        print(f"[TEMPO] {'total'.ljust(width)}  {fmt_seconds(total).rjust(9)}", flush=True)
        if project_folder and os.path.isdir(project_folder):
            data = {"stages": [{"name": n, "seconds": round(s, 2)} for n, s in self.stages],
                    "total_seconds": round(total, 2)}
            try:
                with open(os.path.join(project_folder, "timings.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=1, ensure_ascii=False)
            except Exception as e:
                print(f"[TEMPO] Nao foi possivel salvar timings.json: {e}")


class ClipTimer:
    """Tempo de um corte no enquadramento, separado por parte (leitura, deteccao, escrita, mux)."""

    def __init__(self, label, video_seconds=0.0):
        self.label = label
        self.video_seconds = float(video_seconds or 0.0)
        self.parts = {}
        self.t0 = time.perf_counter()

    @contextmanager
    def measure(self, key):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.parts[key] = self.parts.get(key, 0.0) + (time.perf_counter() - t)

    def finish(self):
        total = time.perf_counter() - self.t0
        parts = dict(self.parts)
        parts["composicao_e_rastreamento"] = max(0.0, total - sum(self.parts.values()))
        speed = (self.video_seconds / total) if total else 0.0
        detail = " | ".join(f"{k} {fmt_seconds(v)}" for k, v in parts.items())
        print(f"[TEMPO] {self.label}: {fmt_seconds(self.video_seconds)} de video em {fmt_seconds(total)} "
              f"({speed:.2f}x tempo real) | {detail}", flush=True)
        return {"clip": self.label, "video_seconds": round(self.video_seconds, 2),
                "total_seconds": round(total, 2),
                "parts": {k: round(v, 2) for k, v in parts.items()}}


def save_clip_timings(project_folder, rows):
    """Salva os tempos dos cortes e imprime a soma por parte."""
    if not rows:
        return
    totals = {}
    for r in rows:
        for k, v in r["parts"].items():
            totals[k] = totals.get(k, 0.0) + v
    video = sum(r["video_seconds"] for r in rows)
    total = sum(r["total_seconds"] for r in rows)
    print(f"[TEMPO] Enquadramento: {len(rows)} corte(s), {fmt_seconds(video)} de video em {fmt_seconds(total)} | " +
          " | ".join(f"{k} {fmt_seconds(v)}" for k, v in totals.items()), flush=True)
    try:
        with open(os.path.join(project_folder, "timings_clips.json"), "w", encoding="utf-8") as f:
            json.dump({"clips": rows, "totals": {k: round(v, 2) for k, v in totals.items()},
                       "video_seconds": round(video, 2), "total_seconds": round(total, 2)}, f, indent=1)
    except Exception as e:
        print(f"[TEMPO] Nao foi possivel salvar timings_clips.json: {e}")
