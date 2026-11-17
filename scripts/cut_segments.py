from scripts import cut_json
from scripts import edit_pauses
import os
import subprocess
import json

def cut(segments, project_folder="tmp", skip_video=False, remove_pauses=False, remove_fillers=False):

    def check_nvenc_support():
        """NVENC listado nao basta (ex.: ffmpeg com NVENC numa maquina sem NVIDIA): testa um encode curto."""
        try:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            if "h264_nvenc" not in result.stdout:
                return False
            test = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                                   "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
                                  capture_output=True, timeout=30)
            return test.returncode == 0
        except Exception:
            return False

    def generate_segments(response, project_folder, skip_video):
        if not check_nvenc_support():
            print("NVENC is not supported on this system. Falling back to libx264.")
            video_codec = "libx264"
        else:
            video_codec = "h264_nvenc"

        # Procurar input_video.mp4 no project_folder ou tmp
        input_file = os.path.join(project_folder, "input.mp4")
        if not os.path.exists(input_file):
            # Tenta fallback legado
            input_file_legacy = os.path.join(project_folder, "input_video.mp4")
            if os.path.exists(input_file_legacy):
                input_file = input_file_legacy
            else:
                print(f"Input file not found in {project_folder}")
                return

        # Pasta de saida para os cortes
        cuts_folder = os.path.join(project_folder, "cuts")
        os.makedirs(cuts_folder, exist_ok=True)
        
        # Pasta de saida para legendas json cortadas
        subs_folder = os.path.join(project_folder, "subs")
        os.makedirs(subs_folder, exist_ok=True)

        # Input JSON (Transkription original)
        input_json_path = os.path.join(project_folder, "input.json")

        segments = response.get("segments", [])
        words = edit_pauses.load_words(project_folder) if (remove_pauses or remove_fillers) else []
        if (remove_pauses or remove_fillers) and not words:
            print("Remoção de pausas/vícios: sem palavras com tempo no input.json; os cortes ficam sem essa edição.")
        for i, segment in enumerate(segments):
            start_time = segment.get("start_time", "00:00:00")
            duration = segment.get("duration", 0)

            # Heurística para duration:
            if isinstance(duration, (int, float)):
                if duration < 1000:
                    duration_seconds = float(duration)
                else:
                    duration_seconds = duration / 1000.0
                duration_str = f"{duration_seconds:.3f}"
            else:
                # Tenta converter string (HH:MM:SS ou float str)
                try:
                    duration_seconds = float(duration)
                    duration_str = f"{duration_seconds:.3f}"
                except ValueError:
                    # Assumindo formato hh:mm:ss se nao for float
                     # Implementar parser se necessario, mas assumindo float por enquanto baseado no historico
                    duration_seconds = 0
                    duration_str = duration
            
            # Heurística para start_time:
            if isinstance(start_time, (int, float)):
                if start_time > 10000: # Se for milisegundos grandes? Assumindo segundos ou HHMMSS?
                   # O código original: if start_time int -> start_time/1000.0.
                   # Vamos manter a lógica original: int -> milisegundos
                   pass
            
            # Refazendo a logica original exata para seguranca e capturando o float:
            if isinstance(start_time, int):
                start_time_seconds = start_time / 1000.0
                start_time_str = f"{start_time_seconds:.3f}"
            elif isinstance(start_time, float):
                 start_time_seconds = start_time
                 start_time_str = f"{start_time_seconds:.3f}"
            else:
                # String "00:00:00" ou "12.34"
                try:
                    start_time_seconds = float(start_time)
                    start_time_str = f"{start_time_seconds:.3f}"
                except:
                    # Se for HH:MM:SS, ffmpeg aceita, mas precisamos converter para float para o json cutter
                    # Função auxiliar simples
                    h, m, s = str(start_time).split(':')
                    start_time_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    start_time_str = str(start_time)

            # Título para nome de arquivo
            title = segment.get("title", f"Segment_{i}")
            safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip()
            safe_title = safe_title.replace(" ", "_")[:60]
            base_name = f"{i:03d}_{safe_title}"

            output_filename = f"{base_name}_original_scale.mp4"
            output_path = os.path.join(cuts_folder, output_filename)

            print(f"Processing segment {i+1}/{len(segments)}")
            print(f"Start time: {start_time}, Duration: {duration}")
            # print(f"Executing command: {' '.join(command)}")

            # VIDEO GENERATION
            removals_path = os.path.join(cuts_folder, f"{base_name}_removals.json")
            removals = []
            if not skip_video:
                if words and duration_seconds:
                    removals = edit_pauses.plan_removals(words, start_time_seconds, start_time_seconds + float(duration_seconds),
                                                         input_file, remove_pauses=remove_pauses, remove_fillers=remove_fillers)
                    if removals:
                        print(f"Pausas/vícios: {len(removals)} trecho(s), {sum(b - a for a, b in removals):.1f}s removidos")
                with open(removals_path, "w", encoding="utf-8") as rf:
                    json.dump(removals, rf)
                # Comando ffmpeg
                command = [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error", "-hide_banner",
                    "-ss", start_time_str,
                    "-i", input_file,
                    "-t", duration_str,
                ]
                if removals:
                    vf, af = edit_pauses.select_filters(edit_pauses.keep_intervals(removals, float(duration_seconds)))
                    command.extend(["-vf", vf, "-af", af])
                command.extend(["-c:v", video_codec])

                if video_codec == "h264_nvenc":
                    command.extend([
                        "-preset", "p1",
                        "-b:v", "5M",
                    ])
                else:
                    command.extend([
                        "-preset", "ultrafast",
                        "-crf", "23"
                    ])

                command.extend([
                    "-c:a", "aac",
                    "-b:a", "128k",
                    output_path
                ])

                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"Generated segment: {output_filename}, Size: {file_size} bytes")
                except subprocess.CalledProcessError as e:
                    print(f"Error executing ffmpeg: {e}")
            else:
                print(f"Skipping video generation for {output_filename} (using existing). check json...")
            
            # --- JSON CUTTING (ALWAYS RUN) ---
            end_time_seconds = start_time_seconds + float(duration_seconds)
            
            # Nome do json correspondente ao vídeo FINAL com titulo
            json_output_filename = f"{base_name}_processed.json"
            json_output_path = os.path.join(subs_folder, json_output_filename)
            
            cut_json.cut_json_transcript(input_json_path, json_output_path, start_time_seconds, end_time_seconds)
            # Cortes com pausas/vicios removidos: remapeia os tempos da legenda para a nova linha do tempo
            if skip_video and os.path.exists(removals_path):
                try:
                    with open(removals_path, "r", encoding="utf-8") as rf:
                        removals = [tuple(r) for r in json.load(rf)]
                except Exception:
                    removals = []
            if removals and os.path.exists(json_output_path):
                with open(json_output_path, "r", encoding="utf-8") as jf:
                    sub_data = json.load(jf)
                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(edit_pauses.remap_subtitle_json(sub_data, removals), jf, indent=2, ensure_ascii=False)
            # --------------------

            print("\n" + "="*50 + "\n")

    # Reading the JSON file if segments not provided (legacy behavior)
    if segments is None:
        json_path = os.path.join(project_folder, 'viral_segments.txt')
        with open(json_path, 'r', encoding='utf-8') as file:
            response = json.load(file)
    else:
        response = segments

    generate_segments(response, project_folder, skip_video)
