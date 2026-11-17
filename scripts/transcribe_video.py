import os
import sys
import torch
import time
import whisperx
import gc
import re
import glob
from i18n.i18n import I18nAuto

i18n = I18nAuto()

def apply_safe_globals_hack():
    """
    Workaround for 'Weights only load failed' error in newer PyTorch versions.
    We first try to add safe globals. If that's not enough/fails, we monkeypatch torch.load.
    """
    try:
        import omegaconf
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                omegaconf.listconfig.ListConfig,
                omegaconf.dictconfig.DictConfig,
                omegaconf.base.ContainerMetadata,
                omegaconf.base.Node
            ])
            print("Aplicado patch de segurança para globals do Omegaconf.")
            
        # Monkeypatch agressivo para garantir compatibilidade com Pyannote/WhisperX antigos
        original_load = torch.load
        
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
            
        torch.load = safe_load
        print("Aplicado monkeypatch em torch.load para forçar weights_only=False.")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Aviso ao tentar aplicar patch de globals: {e}")

    try:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: []
            print("Aplicado monkeypatch em torchaudio.list_audio_backends para PyTorch >= 2.4.")
    except Exception as e:
        pass

def parse_srt(srt_path):
    """
    Parses an SRT file into a list of segments expected by WhisperX alignment.
    [{'start': float, 'end': float, 'text': str}, ...]
    """
    print(f"Parsing SRT: {srt_path}")
    segments = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        content = content.replace('\r\n', '\n')
        blocks = content.strip().split('\n\n')
        
        def time_to_seconds(t_str):
            # SRT: 00:00:00,000
            t_str = t_str.replace(',', '.')
            parts = t_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return 0.0

        for block in blocks:
            lines = block.split('\n')
            # Busca linha de tempo
            for i, line in enumerate(lines):
                if '-->' in line:
                    start_str, end_str = line.split(' --> ')
                    text_lines = lines[i+1:]
                    text = " ".join(text_lines).strip()
                    text = re.sub(r'<[^>]+>', '', text) # Remove tags
                    
                    if text:
                        start = time_to_seconds(start_str.strip())
                        end = time_to_seconds(end_str.strip())
                        segments.append({
                            "start": start,
                            "end": end,
                            "text": text
                        })
                    break
    except Exception as e:
        print(f"Error parsing SRT {srt_path}: {e}")
        return None
    return segments

def parse_vtt(vtt_path):
    """
    Parses a VTT file (WebVTT) into valid segments for WhisperX.
    """
    print(f"Parsing VTT: {vtt_path}")
    segments = []
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        def vtt_time_to_seconds(t_str):
            # VTT: 00:00:00.000 or 00:00.000
            t_str = t_str.strip()
            parts = t_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return 0.0

        current_entry = {"text": []}
        
        for line in lines:
            line = line.strip()
            if not line:
                # Fim de bloco, salva se tiver tempo e texto
                if "start" in current_entry and current_entry["text"]:
                    full_text = " ".join(current_entry["text"]).strip()
                    # Limpeza extra VTT
                    full_text = re.sub(r'<[^>]+>', '', full_text)
                    full_text = re.sub(r'&[^;]+;', '', full_text)
                    
                    if full_text:
                        segments.append({
                            "start": current_entry["start"],
                            "end": current_entry["end"],
                            "text": full_text
                        })
                current_entry = {"text": []}
                continue
            
            if line.startswith("WEBVTT") or line.startswith("X-TIMESTAMP-MAP") or line.startswith("NOTE"):
                continue

            # Timestamp line: 00:00:05.000 --> 00:00:10.000 (pode ter settings depois)
            if "-->" in line:
                times = line.split("-->")
                start_str = times[0].strip()
                end_str = times[1].strip().split(" ")[0] # remove settings
                current_entry["start"] = vtt_time_to_seconds(start_str)
                current_entry["end"] = vtt_time_to_seconds(end_str)
            else:
                # É texto (se já tivermos timestamps)
                if "start" in current_entry:
                     current_entry["text"].append(line)
                     
        # Salva ultimo bloco se existir
        if "start" in current_entry and current_entry["text"]:
            full_text = " ".join(current_entry["text"]).strip()
            full_text = re.sub(r'<[^>]+>', '', full_text)
            if full_text:
                segments.append({
                    "start": current_entry["start"],
                    "end": current_entry["end"],
                    "text": full_text
                })

    except Exception as e:
        print(f"Error parsing VTT {vtt_path}: {e}")
        return None
    return segments

_VTT_TS = r'(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})'


def clean_caption_text(text):
    """Limpa texto de legenda do YouTube: tags, entidades HTML (&gt;&gt;), marcadores >> e [Música]."""
    import html
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = text.replace('>>', ' ')
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _ts_to_seconds(t_str):
    parts = t_str.strip().replace(',', '.').split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def parse_youtube_vtt_words(vtt_path):
    """
    Le a legenda automatica do YouTube (formato karaoke) com o tempo de cada palavra:
    "texto<00:00:01.120><c> palavra</c><00:00:01.400><c> outra</c>".
    So as linhas com marcacao de tempo trazem palavras novas (as outras repetem a linha anterior).
    Retorna (segments, idioma) ou (None, idioma) se a legenda nao tiver tempo por palavra.
    """
    language = None
    segments = []
    cue_start = cue_end = None
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading VTT {vtt_path}: {e}")
        return None, None

    for raw in lines:
        line = raw.strip()
        if line.startswith('Language:'):
            language = line.split(':', 1)[1].strip() or None
            continue
        if '-->' in line:
            a, b = line.split('-->')
            cue_start = _ts_to_seconds(a)
            cue_end = _ts_to_seconds(b.strip().split(' ')[0])
            continue
        if cue_start is None or not re.search('<' + _VTT_TS + '>', line):
            continue
        parts = re.split('<' + _VTT_TS + '>', line)
        times = [cue_start] + [_ts_to_seconds(t) for t in parts[1::2]]
        words = []
        for t, chunk in zip(times, parts[0::2]):
            for w in clean_caption_text(chunk).split():
                words.append({"word": w, "start": round(t, 3)})
        if not words:
            continue
        for i, w in enumerate(words):
            nxt = words[i + 1]["start"] if i + 1 < len(words) else min(cue_end, w["start"] + 1.0)
            w["end"] = round(max(nxt, w["start"] + 0.05), 3)
            w["score"] = 1.0
        segments.append({
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),
            "words": words,
        })
    return (segments or None), language


def add_estimated_words(segments):
    """Legenda sem tempo por palavra: distribui o tempo do trecho entre as palavras pelo tamanho."""
    out = []
    for seg in segments:
        text = clean_caption_text(seg.get("text", ""))
        tokens = text.split()
        if not tokens:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        total = sum(len(t) + 1 for t in tokens)
        words, t = [], start
        for tok in tokens:
            dur = (end - start) * (len(tok) + 1) / total
            words.append({"word": tok, "start": round(t, 3), "end": round(t + dur, 3), "score": 1.0})
            t += dur
        out.append({"start": start, "end": end, "text": text, "words": words})
    return out


def pick_batch_size(device):
    """Escolhe o batch do Whisper pela VRAM total. Env VIRALCUTTER_WHISPER_BATCH força um valor."""
    forced = os.environ.get("VIRALCUTTER_WHISPER_BATCH")
    if forced:
        try:
            return max(1, int(forced))
        except ValueError:
            print(f"VIRALCUTTER_WHISPER_BATCH inválido ({forced}); usando automático.")
    if device != "cuda":
        return 4
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 40:
        return 64
    if total_gb >= 20:
        return 48
    if total_gb >= 14:
        return 32
    if total_gb >= 8:
        return 16
    return 8


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def transcribe(input_file, model_name='large-v3', project_folder='tmp', batch_size=None):
    print(i18n(f"Iniciando transcrição de {input_file}..."))
    
    # Diagnóstico de Ambiente
    print(f"DEBUG: Python: {sys.executable}")
    print(f"DEBUG: Torch: {torch.__version__}")
    
    start_time = time.time()
    
    if project_folder is None:
        project_folder = os.path.dirname(input_file)
        if not project_folder:
            project_folder = 'tmp'

    output_folder = project_folder
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    srt_file = os.path.join(output_folder, f"{base_name}.srt")
    tsv_file = os.path.join(output_folder, f"{base_name}.tsv")
    json_file = os.path.join(output_folder, f"{base_name}.json")

    # Verifica se os arquivos já existem
    if os.path.exists(srt_file) and os.path.exists(tsv_file) and os.path.exists(json_file):
        print(f"Os arquivos SRT, TSV e JSON já existem. Pulando a transcrição.")
        return srt_file, tsv_file

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: Usando dispositivo: {device}")
    # CPU: int8 usa ~metade da RAM do float32 (pico ~2 GB com batch 4 no large-v3-turbo)
    compute_type = "float16" if device == "cuda" else "int8"
    if batch_size is None:
        batch_size = pick_batch_size(device)

    try:
        apply_safe_globals_hack()
        
        # 1. Carregar Áudio (sempre necessário)
        print(f"Carregando áudio: {input_file}")
        audio = whisperx.load_audio(input_file)
        
        # 2. Legenda do YouTube: usa os tempos da propria legenda e pula transcricao e alinhamento.
        # input.vtt (automatica, com tempo por palavra) tem prioridade sobre input.srt.
        vtt_path = os.path.join(output_folder, "input.vtt")
        srt_path = os.path.join(output_folder, "input.srt")

        start_segments = None
        alignment_only = False  # True = legenda pronta, sem transcrever nem alinhar
        detected_language = "und"

        if os.path.exists(vtt_path):
            print(f"Usando legenda do YouTube: {vtt_path}")
            parsed, sub_lang = parse_youtube_vtt_words(vtt_path)
            if parsed:
                print(f"Legenda com tempo por palavra: {len(parsed)} trechos.")
            else:
                parsed = parse_vtt(vtt_path)
                parsed = add_estimated_words(parsed) if parsed else None
            if parsed:
                start_segments, alignment_only = parsed, True
                detected_language = sub_lang or detected_language
        if not alignment_only and os.path.exists(srt_path):
            print(f"Usando legenda do YouTube: {srt_path}")
            parsed = parse_srt(srt_path)
            if parsed:
                start_segments, alignment_only = add_estimated_words(parsed), True

        if alignment_only:
            print("--- LEGENDA DO YOUTUBE: sem transcrição e sem alinhamento ---")

        result = None
        
        if alignment_only and start_segments:
            # Legenda pronta: pula a transcrição
            # Estrutura que o align espera: {'segments': [...], 'language': ...}
            # Mas o align recebe segments como lista.
            pass 
        else:
            # 3. Transcrever (Caminho Normal)
            print("Nenhuma legenda válida encontrada. Realizando transcrição completa (WhisperX)...")
            print(f"Carregando modelo {model_name}...")
            model = whisperx.load_model(
                model_name, 
                device, 
                compute_type=compute_type,
                asr_options={"hotwords": None}
            )

            while True:
                try:
                    print(f"Transcrevendo com batch_size={batch_size} ({compute_type})...")
                    result = model.transcribe(
                        audio, 
                        batch_size=batch_size, 
                        chunk_size=10
                    )
                    break
                except (RuntimeError, MemoryError) as e:
                    if device == "cuda" and _is_oom(e) and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"Sem VRAM suficiente; tentando de novo com batch_size={batch_size}.")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    raise
            
            detected_language = result["language"]
            start_segments = result["segments"]
            
            # Limpar modelo de transcrição
            if device == "cuda":
                del model
                gc.collect()
                torch.cuda.empty_cache()

        # 4. Alinhar (Sempre executado, seja com subs parsed ou transcritos)
        if alignment_only:
            result = {"segments": start_segments, "language": detected_language}
        else:
            print(f"Alinhando transcrição (Idioma: {detected_language}) para obter timestamps precisos...")
            # Usa o modelo específico solicitado pelo usuário: WAV2VEC2_ASR_LARGE_LV60K_960H
            # Mas o whisperx.load_align_model escolhe automaticamente baseado na linguagem.
            # Se for inglês, ele usa wav2vec2-large-960h-lv60-self geralmente.
            # Não podemos forçar facilmente o modelo exato sem hackear o whisperx, mas o padrão é bom.
        
            try:
                model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
            
                aligned_result = whisperx.align(start_segments, model_a, metadata, audio, device, return_char_alignments=False)
            
                # aligned_result agora contém "segments" com word timestamps
                result = aligned_result
                result["language"] = detected_language
            
                if device == "cuda":
                     del model_a
                     torch.cuda.empty_cache()
                 
            except Exception as e:
                print(f"Erro durante alinhamento: {e}. ")
                if alignment_only:
                     print("Falha crítica no alinhamento de legendas externas. Abortando usage de legendas externas.")
                     # Opcional: Fallback para transcrição normal se falhar? Seria complexo aqui pois já limpamos memória.
                     # Vamos apenas salvar o que temos (timestamps da legenda original podem não bater com áudio perfeitamente se não alinhar)
                     result = {"segments": start_segments, "language": detected_language}
                else:
                     print("Continuando com transcrição bruta.")

        # 5. Salvar Resultados
        print("Salvando resultados...")
        from whisperx.utils import get_writer
        
        save_options = {
            "highlight_words": False,
            "max_line_count": None,
            "max_line_width": None
        }
        
        # Se veio do alignment_only, result é {'segments': [...], ...}
        # Se o alinhamento falhou, result tem segments originais.
        
        # WhisperX writers esperam um dicionário result com chaves 'segments', 'language'.
        
        writer_srt = get_writer("srt", output_folder)
        writer_srt(result, input_file, save_options)
        
        writer_tsv = get_writer("tsv", output_folder)
        writer_tsv(result, input_file, save_options)
        
        writer_json = get_writer("json", output_folder)
        writer_json(result, input_file, save_options)
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Processamento concluído em {int(elapsed//60)}m {int(elapsed%60)}s.")

    except Exception as e:
        print(f"ERRO CRÍTICO na transcrição: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not os.path.exists(srt_file):
        print(f"AVISO: Arquivo SRT {srt_file} não encontrado após execução.")
    
    return srt_file, tsv_file
