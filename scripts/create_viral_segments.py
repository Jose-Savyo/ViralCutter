import json
import os
import re
import sys
import time
import ast
import io

# Configura stdout para evitar erros de encoding no Windows (substitui caracteres inválidos por ?)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        # Mantém encoding original mas ignora erros (substitui por ?)
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or 'utf-8', errors='replace', line_buffering=True)
    except:
        pass

# Tenta importar bibliotecas de IA opcionalmente
# Gemini: SDK novo (google-genai) com timeout e controle de raciocinio; o antigo
# (google-generativeai, descontinuado) so e usado se o novo nao estiver instalado.
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

genai = None
HAS_GEMINI_LEGACY = False
if not HAS_GENAI:
    try:
        import google.generativeai as genai
        HAS_GEMINI_LEGACY = True
    except ImportError:
        pass
HAS_GEMINI = HAS_GENAI or HAS_GEMINI_LEGACY

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

try:
    import g4f
    HAS_G4F = True
except ImportError:
    HAS_G4F = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Modelos/URLs padrao dos provedores adicionais
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"

def clean_json_response(response_text):
    """
    Limpa a resposta focando em encontrar o objeto JSON que contém a chave "segments".
    Estratégia: 
    1. Busca a palavra "segments", encontra o '{' anterior e usa raw_decode.
    2. Fallback: Parsear lista de segmentos item a item (recuperação de JSON truncado).
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    if not response_text:
        return {"segments": []}

    # 1. Limpeza preliminar
    # Remove tags de pensamento (DeepSeek R1)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # Normaliza escapes excessivos (\n virando \\n) e aspas se parecer necessário
    try:
        if "\\n" in response_text or "\\\"" in response_text:
             # Tenta um decode básico de escapes
             response_text = response_text.replace("\\n", "\n").replace("\\\"", "\"").replace("\\'", "'")
    except:
        pass

    # 2. Busca pela palavra-chave "segments"
    # Procura índices de todas as ocorrências de 'segments'
    matches = [m.start() for m in re.finditer(r'segments', response_text)]
    
    if not matches:
        # Se não achou segments, retorna vazio
        return {"segments": []}

    # Tenta extrair JSON válido a partir de cada ocorrência
    for match_idx in matches:
        # Procura o '{' mais próximo ANTES de "segments"
        # Limita busca a 5000 chars para trás para performance
        start_search = max(0, match_idx - 5000)
        snippet_before = response_text[start_search:match_idx]
        
        # Encontra o ÚLTIMO '{' no snippet
        last_open_rel = snippet_before.rfind('{')
        
        if last_open_rel != -1:
            real_start = start_search + last_open_rel
            candidate_text = response_text[real_start:]
            
            # Tentativa A: json.raw_decode
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(candidate_text)
                if 'segments' in obj and isinstance(obj['segments'], list):
                    return obj
            except:
                pass
            
            # Tentativa B: ast.literal_eval
            try:
                balance = 0
                in_string = False
                escape = False
                found_end = -1
                
                for i, char in enumerate(candidate_text):
                    if escape:
                        escape = False
                        continue
                    if char == '\\':
                        escape = True
                        continue
                    if char == "'" or char == '"':
                        in_string = not in_string
                        continue
                        
                    if not in_string:
                        if char == '{':
                            balance += 1
                        elif char == '}':
                            balance -= 1
                            if balance == 0:
                                found_end = i
                                break
                
                if found_end != -1:
                    clean_cand = candidate_text[:found_end+1]
                    obj = ast.literal_eval(clean_cand)
                    if 'segments' in obj and isinstance(obj['segments'], list):
                        return obj
            except:
                pass

    # 3. Fallback: Extração bruta de markdown
    try:
        match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except:
        pass
        
    # 3b. json-repair: conserta virgulas/aspas faltando e JSON truncado
    if HAS_JSON_REPAIR:
        try:
            fixed = repair_json(response_text, return_objects=True)
            if isinstance(fixed, list):
                fixed = next((x for x in fixed if isinstance(x, dict) and "segments" in x), None)
            if isinstance(fixed, dict) and isinstance(fixed.get("segments"), list):
                return fixed
        except Exception:
            pass

    # 4. LAST RESORT: Fragment Parser (Para JSON truncado/incompleto)
    # Procura por "segments": [ e tenta parsear item por item
    try:
        match_list = re.search(r'"segments"\s*:\s*\[', response_text)
        if match_list:
            start_pos = match_list.end()
            current_pos = start_pos
            found_segments = []
            decoder = json.JSONDecoder()
            
            while True:
                while current_pos < len(response_text) and response_text[current_pos] in ' \t\n\r,':
                    current_pos += 1
                
                if current_pos >= len(response_text):
                    break
                    
                if response_text[current_pos] == ']':
                    break
                
                try:
                    obj, end_pos = decoder.raw_decode(response_text[current_pos:])
                    if isinstance(obj, dict):
                        found_segments.append(obj)
                    current_pos += end_pos
                except json.JSONDecodeError:
                    break
                    
            if found_segments:
                print(f"[INFO] Recuperado {len(found_segments)} segmentos de JSON truncado.")
                return {"segments": found_segments}
    except:
        pass

    return {"segments": []}


def preprocess_transcript_for_ai(segments):
    """
    Concatenates transcript segments into a single string with embedded time tags.
    """
    if not segments:
        return ""

    full_text = ""
    last_tag_time = -100  # Force first tag
    
    # Try to start with (0s) based on first segment
    first_start = segments[0].get('start', 0)
    full_text += f"({int(first_start)}s) "
    last_tag_time = first_start

    for seg in segments:
        text = seg.get('text', '').strip()
        end_time = seg.get('end', 0)
        
        full_text += text + " "
        
        if end_time - last_tag_time >= 4:
            full_text += f"({int(end_time)}s) "
            last_tag_time = end_time

    return full_text.strip()

# --- Robustez das chamadas de IA ---
AI_TIMEOUT_SECONDS = int(os.environ.get("VIRALCUTTER_AI_TIMEOUT", "180"))
QUOTA_MAX_WAIT_SECONDS = 120          # cota "por minuto" com espera maior que isso e tratada como esgotada
TRANSIENT_HTTP_CODES = {408, 500, 502, 503, 504, 529}
AI_STATS = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
_GEMINI_NO_THINKING = set()           # modelos que recusaram thinking_config nesta execucao


class AIQuotaError(RuntimeError):
    """Cota da API esgotada (diaria, ou com espera longa demais para valer a pena aguardar)."""


def parse_retry_seconds(text):
    """Tempo de espera pedido pela API: 'retry in 6h7m47.2s', 'retry in 47s', retry_delay { seconds: 22067 }."""
    text = re.sub(r"(\d+(?:\.\d+)?)ms\b", lambda m: f"{float(m.group(1)) / 1000:.3f}s", str(text or ""))
    m = re.search(r"retry in\s+([\d.hms\s]+)", text, re.I)
    if m:
        total = sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)))
        if total > 0:
            return total
    m = re.search(r"retry_?delay['\"]?\s*[:=]?\s*\{?\s*['\"]?seconds['\"]?\s*[:=]\s*['\"]?(\d+)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", text)
    if m:
        return float(m.group(1))
    return None


def is_daily_quota(text):
    return bool(re.search(r"per\s*day|PerDay|daily", str(text or ""), re.I))


def _fmt_wait(seconds):
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}min" if h else f"{max(1, rem // 60)}min"


def quota_message(provider, model, wait_seconds, daily):
    kind = "diária" if daily else "por minuto"
    when = f" Libera em ~{_fmt_wait(wait_seconds)}." if wait_seconds else ""
    return (f"Cota {kind} da API esgotada ({provider}, modelo {model}).{when} "
            f"Opções: usar outro modelo (ex.: gemini-flash-lite-latest), outro provedor "
            f"(--ai-backend openai ou anthropic) ou esperar a cota liberar.")


if HAS_GENAI:
    from pydantic import BaseModel as _BaseModel

    class _SegmentOut(_BaseModel):
        start_text: str
        end_text: str
        start_time_ref: str
        title: str
        hook: str
        reasoning: str
        score: int

    class _SegmentsOut(_BaseModel):
        segments: list[_SegmentOut]

    SEGMENTS_RESPONSE_SCHEMA = _SegmentsOut
else:
    SEGMENTS_RESPONSE_SCHEMA = None


def _gemini_thinking_config(model_name):
    """Raciocinio baixo por padrao (VIRALCUTTER_GEMINI_THINKING=minimal|low|medium|high|<orcamento>|default)."""
    if not HAS_GENAI or model_name in _GEMINI_NO_THINKING:
        return None
    raw = os.environ.get("VIRALCUTTER_GEMINI_THINKING", "low").strip().lower()
    if raw in ("", "default", "auto"):
        return None
    name = (model_name or "").lower()
    if raw.isdigit():
        return genai_types.ThinkingConfig(thinking_budget=int(raw))
    if "1.5" in name or "2.0" in name:
        return None  # modelos sem raciocinio
    if "2.5" in name:
        budget = {"minimal": 0, "low": 1024, "medium": 4096, "high": 8192}.get(raw, 1024)
        if "pro" in name:
            budget = max(budget, 128)  # o 2.5 pro nao aceita desligar o raciocinio
        return genai_types.ThinkingConfig(thinking_budget=budget)
    level = raw.upper() if raw.upper() in ("MINIMAL", "LOW", "MEDIUM", "HIGH") else "LOW"
    return genai_types.ThinkingConfig(thinking_level=level)


def _call_gemini_genai(prompt, api_key, model_name, response_schema=None):
    client = google_genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(
        timeout=AI_TIMEOUT_SECONDS * 1000, retry_options=genai_types.HttpRetryOptions(attempts=1)))
    cfg = {"response_mime_type": "application/json"}
    if response_schema is not None:
        cfg["response_schema"] = response_schema
    thinking = _gemini_thinking_config(model_name)
    if thinking is not None:
        cfg["thinking_config"] = thinking

    max_attempts, attempt, timeouts = 5, 0, 0
    while attempt < max_attempts:
        attempt += 1
        AI_STATS["requests"] += 1
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt,
                                                  config=genai_types.GenerateContentConfig(**cfg))
        except genai_errors.APIError as e:
            code = getattr(e, "code", 0) or 0
            text = str(e)
            if code == 429:
                wait = parse_retry_seconds(text)
                daily = is_daily_quota(text)
                if daily or (wait and wait > QUOTA_MAX_WAIT_SECONDS):
                    raise AIQuotaError(quota_message("Gemini", model_name, wait, daily))
                wait = (wait + 2) if wait else 20 * attempt
                print(f"[429] Limite por minuto do Gemini ({model_name}). Aguardando {wait:.0f}s "
                      f"({attempt}/{max_attempts})...", flush=True)
                time.sleep(wait)
                continue
            if code == 400 and "thinking" in text.lower() and "thinking_config" in cfg:
                print(f"[IA] {model_name} não aceita configuração de raciocínio; repetindo sem ela.", flush=True)
                cfg.pop("thinking_config")
                _GEMINI_NO_THINKING.add(model_name)
                attempt -= 1
                continue
            if code == 400 and "schema" in text.lower() and "response_schema" in cfg:
                print(f"[IA] {model_name} recusou o esquema de resposta; repetindo só com JSON.", flush=True)
                cfg.pop("response_schema")
                attempt -= 1
                continue
            if code in TRANSIENT_HTTP_CODES:
                wait = min(60, 5 * 2 ** (attempt - 1))
                print(f"[{code}] Gemini indisponível no momento. Nova tentativa em {wait}s "
                      f"({attempt}/{max_attempts})...", flush=True)
                time.sleep(wait)
                continue
            print(f"Erro na API do Gemini (HTTP {code}): {getattr(e, 'message', None) or text[:300]}")
            return "{}"
        except Exception as e:
            msg = str(e)
            if "timed out" in msg.lower() or "timeout" in type(e).__name__.lower():
                timeouts += 1
                if timeouts >= 2:
                    print(f"[IA] Gemini não respondeu em {AI_TIMEOUT_SECONDS}s (2 vezes). Desistindo deste trecho.", flush=True)
                    return "{}"
                print(f"[IA] Gemini não respondeu em {AI_TIMEOUT_SECONDS}s. Tentando de novo...", flush=True)
                continue
            print(f"Erro ao chamar o Gemini: {e}")
            return "{}"

        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            t_in = getattr(usage, "prompt_token_count", 0) or 0
            t_out = getattr(usage, "candidates_token_count", 0) or 0
            t_think = getattr(usage, "thoughts_token_count", 0) or 0
            AI_STATS["input_tokens"] += t_in
            AI_STATS["output_tokens"] += t_out
            AI_STATS["thinking_tokens"] += t_think
            print(f"[IA] Gemini respondeu em {time.time() - t0:.1f}s | tokens: entrada {t_in}, "
                  f"saída {t_out}, raciocínio {t_think}", flush=True)
        try:
            text = resp.text
        except Exception:
            text = None
        if not text:
            reason = None
            try:
                reason = resp.candidates[0].finish_reason
            except Exception:
                pass
            print(f"[WARN] Resposta vazia do Gemini (motivo: {reason}).")
            return "{}"
        return text

    print(f"Falha após {max_attempts} tentativas no Gemini.")
    return "{}"


def _call_gemini_legacy(prompt, api_key, model_name):
    """SDK antigo (google-generativeai): so quando o google-genai nao esta instalado."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
    max_retries = 5
    for attempt in range(max_retries):
        AI_STATS["requests"] += 1
        try:
            response = model.generate_content(prompt, request_options={"timeout": AI_TIMEOUT_SECONDS, "retry": None})
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str or "ResourceExhausted" in type(e).__name__:
                wait = parse_retry_seconds(error_str)
                daily = is_daily_quota(error_str)
                if daily or (wait and wait > QUOTA_MAX_WAIT_SECONDS):
                    raise AIQuotaError(quota_message("Gemini", model_name, wait, daily))
                wait = (wait + 2) if wait else 30 * (attempt + 1)
                print(f"[429] Limite por minuto do Gemini. Aguardando {wait:.0f}s ({attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
                continue
            print(f"Erro na API do Gemini: {e}")
            return "{}"
    print("Falha após max retries no Gemini.")
    return "{}"


def call_gemini(prompt, api_key, model_name='gemini-flash-lite-latest', response_schema=None):
    if HAS_GENAI:
        return _call_gemini_genai(prompt, api_key, model_name, response_schema)
    if HAS_GEMINI_LEGACY:
        return _call_gemini_legacy(prompt, api_key, model_name)
    raise ImportError("Nenhum SDK do Gemini instalado. Instale com: pip install google-genai")

def call_g4f(prompt, model_name=None):
    if not HAS_G4F:
        raise ImportError("A biblioteca 'g4f' não está instalada. Instale com: pip install g4f")
    if not model_name:
        model_name = g4f.models._all_models[0]
    
    max_retries = 3
    base_wait = 5
    
    for attempt in range(max_retries):
        try:
            AI_STATS["requests"] += 1
            response = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            
            if isinstance(response, dict):
                if 'error' in response:
                    raise Exception(f"API Error: {response['error']}")
                if 'choices' in response and isinstance(response['choices'], list):
                    if len(response['choices']) > 0:
                         content = response['choices'][0].get('message', {}).get('content', '')
                         if content:
                             return content
                if not response:
                     raise ValueError("Empty Dict response")

                return json.dumps(response)

            if not response:
                print(f"[WARN] G4F retornou resposta vazia. Tentativa {attempt+1}/{max_retries}")
                time.sleep(base_wait)
                continue
            
            if isinstance(response, str):
                return response

            try:
                return json.dumps(response, ensure_ascii=False)
            except:
                return str(response)
            
        except Exception as e:
            print(f"[WARN] Erro na API do G4F (Tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                time.sleep(wait_time)
            
    print(f"Falha crítica após {max_retries} tentativas no G4F.")
    return "{}"

def call_openai_compatible(prompt, api_key, model_name=None, base_url=None):
    """
    Qualquer API compativel com o endpoint /chat/completions da OpenAI:
    OpenAI, OpenRouter, Groq, DeepSeek, Ollama (http://localhost:11434/v1), LM Studio, etc.
    """
    import requests

    base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    model_name = model_name or DEFAULT_OPENAI_MODEL
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:  # servidores locais (Ollama/LM Studio) nao exigem chave
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}

    max_retries = 5
    for attempt in range(max_retries):
        try:
            AI_STATS["requests"] += 1
            resp = requests.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            wait_time = 5 * (attempt + 1)
            print(f"[WARN] Erro de conexão com {base_url} ({e}). Nova tentativa em {wait_time}s ({attempt+1}/{max_retries})...", flush=True)
            time.sleep(wait_time)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("retry-after")
            try:
                wait_time = float(retry_after) + 1.0 if retry_after else 15.0 * (attempt + 1)
            except ValueError:
                wait_time = 15.0 * (attempt + 1)
            print(f"[{resp.status_code}] {base_url} ocupado/limite. Aguardando {wait_time:.0f}s ({attempt+1}/{max_retries})...", flush=True)
            time.sleep(wait_time)
            continue

        if resp.status_code != 200:
            print(f"Erro na API ({base_url}) HTTP {resp.status_code}: {resp.text[:500]}")
            return "{}"

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"] or "{}"
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"Resposta inesperada de {base_url}: {e} | {resp.text[:300]}")
            return "{}"

    print(f"Falha após {max_retries} tentativas em {base_url}.")
    return "{}"

def call_anthropic(prompt, api_key, model_name=None):
    """Claude via SDK oficial da Anthropic (pip install anthropic)."""
    if not HAS_ANTHROPIC:
        raise ImportError("A biblioteca 'anthropic' não está instalada. Instale com: pip install anthropic")

    model_name = model_name or DEFAULT_ANTHROPIC_MODEL
    # api_key=None -> o SDK usa ANTHROPIC_API_KEY / perfil do `ant auth login`
    client = anthropic.Anthropic(api_key=api_key or None, max_retries=5)

    try:
        # fallbacks="default": se o modelo recusar por politica de seguranca, a propria API
        # tenta de novo em um modelo alternativo dentro da mesma chamada.
        AI_STATS["requests"] += 1
        response = client.beta.messages.create(
            model=model_name,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        print("Erro na API da Anthropic: chave inválida.")
        return "{}"
    except anthropic.NotFoundError as e:
        print(f"Erro na API da Anthropic: modelo '{model_name}' não encontrado ({e.message}).")
        return "{}"
    except anthropic.RateLimitError:
        print("Erro na API da Anthropic: limite de uso atingido (429) mesmo após as tentativas.")
        return "{}"
    except anthropic.APIStatusError as e:
        print(f"Erro na API da Anthropic (HTTP {e.status_code}): {e.message}")
        return "{}"
    except anthropic.APIConnectionError as e:
        print(f"Erro de conexão com a API da Anthropic: {e}")
        return "{}"

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        print(f"A Anthropic recusou o pedido (categoria: {category}).")
        return "{}"
    if response.stop_reason == "max_tokens":
        print("[WARN] Resposta da Anthropic cortada em max_tokens; tentando aproveitar o JSON parcial.")

    text = "".join(block.text for block in response.content if block.type == "text")
    return text or "{}"

def load_transcript(project_folder):
    """Parses input.tsv or input.srt from the project folder."""
    input_tsv = os.path.join(project_folder, 'input.tsv')
    input_srt = os.path.join(project_folder, 'input.srt')

    transcript_segments = []
    
    # Try to load TSV first (more reliable time)
    if os.path.exists(input_tsv):
        try:
            with open(input_tsv, 'r', encoding='utf-8') as f:
                # Skip header
                lines = f.readlines()[1:] 
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start_ms = float(parts[0])
                        end_ms = float(parts[1])
                        text = parts[2]
                        transcript_segments.append({
                            'start': start_ms / 1000.0, 
                            'end': end_ms / 1000.0, 
                            'text': text
                        })
        except Exception as e:
            print(f"Error parsing TSV: {e}")

    # Fallback to SRT parser if TSV empty/failed
    if not transcript_segments and os.path.exists(input_srt):
         with open(input_srt, 'r', encoding='utf-8') as f:
             srt_content = f.read()
         pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*)', re.DOTALL)
         matches = pattern.findall(srt_content)
         
         def srt_time_to_seconds(t_str):
             h, m, s = t_str.replace(',', '.').split(':')
             return int(h) * 3600 + int(m) * 60 + float(s)

         for m in matches:
             start_sec = srt_time_to_seconds(m[1])
             end_sec = srt_time_to_seconds(m[2])
             text = m[3].replace('\n', ' ')
             transcript_segments.append({'start': start_sec, 'end': end_sec, 'text': text})

    if not transcript_segments:
        raise ValueError("Could not parse transcript from TSV or SRT.")
    
    return transcript_segments

SENTENCE_END_RE = re.compile(r'[.!?…]["\'”»)\]]*$')

def snap_end(start, desired_end, transcript_segments, min_duration, max_duration):
    """
    Escolhe o fim do corte em um limite de fala da transcricao, dentro de
    [start + min_duration, start + max_duration], preferindo fins de frase (. ? !).
    Retorna (end, motivo): "ok" (fim pedido ja estava dentro dos limites),
    "antes-do-max", "depois-do-min", "sem-candidatos".
    """
    lo, hi = start + min_duration, start + max_duration
    if desired_end is not None and lo <= desired_end <= hi:
        return desired_end, "ok"

    eps = 0.05
    candidates = [s for s in transcript_segments if lo - eps <= s['end'] <= hi + eps]
    if not candidates:
        # Nenhum limite de fala dentro da janela (falas muito longas): mantem o comportamento antigo
        if desired_end is None:
            return lo, "sem-candidatos"
        return min(max(desired_end, lo), hi), "sem-candidatos"

    sentence_ends = [s for s in candidates if SENTENCE_END_RE.search(s['text'].strip())]
    pool = sentence_ends or candidates
    if desired_end is not None and desired_end > hi:
        # Longo demais: ultimo fim de frase antes do maximo
        return max(s['end'] for s in pool), "antes-do-max"
    # Curto demais ou fim nao encontrado: primeiro fim de frase depois do minimo
    return min(s['end'] for s in pool), "depois-do-min"

HOOK_MAX_WORDS = 10


def _clean_text(text):
    """Remove entidades HTML (&gt;&gt;) e marcadores de troca de falante (>>) vindos de legendas."""
    import html
    text = html.unescape(str(text or "")).replace(">>", " ")
    return re.sub(r"\s+", " ", text).strip()


def final_hook(seg, start, end, transcript_segments):
    """Texto curto do inicio do corte, diferente do titulo. Sem hook valido da IA, usa a 1a fala do trecho."""
    title = (seg.get("title") or "").strip()
    hook = _clean_text(seg.get("hook"))
    if hook and hook.lower() != title.lower():
        words = hook.split()
        return " ".join(words[:HOOK_MAX_WORDS + 2])
    text = _clean_text(" ".join((s.get("text") or "") for s in transcript_segments
                                if start - 0.05 <= s["start"] < min(end, start + 10)))
    words = text.split()
    if not words:
        return title
    return " ".join(words[:HOOK_MAX_WORDS]) + ("..." if len(words) > HOOK_MAX_WORDS else "")


def heuristic_segments(transcript_segments, count, min_duration, max_duration, heatmap=None):
    """Reserva quando a IA nao devolve nenhum corte: janelas com mais 'mais repetidos' e mais fala."""
    segs = transcript_segments or []
    if not segs:
        return {"segments": []}
    target = (min_duration + max_duration) / 2.0
    candidates, last_start = [], -1e9
    for s in segs:
        st = s["start"]
        if st - last_start < 5.0:      # uma janela a cada ~5 s basta
            continue
        last_start = st
        end, _ = snap_end(st, st + target, segs, min_duration, max_duration)
        text = _clean_text(" ".join(x["text"] for x in segs if st <= x["start"] < end))
        heat = heat_average(heatmap, st, end) if heatmap else None
        density = len(text.split()) / max(1.0, end - st)
        score = (100.0 * heat if heat is not None else 0.0) + 10.0 * density + 2.0 * (text.count("?") + text.count("!"))
        candidates.append((score, st, end, text))
    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen = []
    for c in candidates:
        if any(c[1] < o[2] and c[2] > o[1] for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= count:
            break
    out = []
    for score, st, end, text in chosen:
        words = text.split()
        mm, ss = divmod(int(st), 60)
        out.append({"title": f"Trecho {mm:02d}:{ss:02d}", "start_time": st, "end_time": end,
                    "hook": " ".join(words[:HOOK_MAX_WORDS]) + ("..." if len(words) > HOOK_MAX_WORDS else ""),
                    "reasoning": "heurística (a IA não devolveu cortes)", "score": int(min(100, score)),
                    "duration": end - st})
    return {"segments": out}


def process_segments(raw_segments, transcript_segments, min_duration, max_duration, output_count=None, heatmap=None):
    """
    Aligns raw AI segments (with reference tags) to actual transcript timestamps.
    Applies constraints, validation, and deduplication.
    """
    
    all_segments = raw_segments
    tempo_minimo = min_duration
    tempo_maximo = max_duration
    
    # Sort segments by score (descending)
    try:
        all_segments.sort(key=lambda x: int(x.get('score', 0)), reverse=True)
    except:
        pass

    # --- POST-PROCESSING: Match Text to Timestamps ---
    processed_segments = []
    
    print(f"[DEBUG] Matching {len(all_segments)} raw segments to timestamps...")
    
    for seg in all_segments:
        try:
            # 1. Parse Reference Time
            ref_time_str = seg.get('start_time_ref', '(0s)')
            ref_time_val = 0
            try:
                if isinstance(ref_time_str, str):
                    match = re.search(r'\d+', ref_time_str)
                    if match:
                         ref_time_val = int(match.group())
                else:
                    ref_time_val = int(ref_time_str)
            except:
                ref_time_val = 0
                
            # Find segment index closest to ref_time
            start_idx = 0
            min_diff = 999999
            for i, s in enumerate(transcript_segments):
                diff = abs(s['start'] - ref_time_val)
                if diff < min_diff:
                    min_diff = diff
                    start_idx = i
                if s['start'] > ref_time_val + 10: 
                    break
            
            # Backtrack
            start_idx = max(0, start_idx - 5)
            
            # 2. Find Exact Start Text
            start_text_target = seg.get('start_text', '').lower().strip()
            # Normalize
            start_text_target = re.sub(r'[^\w\s]', '', start_text_target)
            
            final_start_time = -1
            match_start_idx = -1
            
            # Search window
            search_limit = min(len(transcript_segments), start_idx + 50)
            
            for i in range(start_idx, search_limit):
                s_text = transcript_segments[i]['text'].lower()
                s_text = re.sub(r'[^\w\s]', '', s_text)
                
                # Check for partial match
                if start_text_target and (start_text_target in s_text or s_text in start_text_target):
                    final_start_time = transcript_segments[i]['start']
                    match_start_idx = i
                    break
            
            # Fallback
            if final_start_time == -1:
                final_start_time = transcript_segments[start_idx]['start'] if start_idx < len(transcript_segments) else ref_time_val
                match_start_idx = start_idx

            # 3. Find End Text
            end_text_target = seg.get('end_text', '').lower().strip()
            end_text_target = re.sub(r'[^\w\s]', '', end_text_target)
            
            final_end_time = -1
            
            if match_start_idx != -1:
                search_end_limit = min(len(transcript_segments), match_start_idx + 200)
                
                for i in range(match_start_idx, search_end_limit):
                    s_text = transcript_segments[i]['text'].lower()
                    s_text = re.sub(r'[^\w\s]', '', s_text)
                    
                    if end_text_target and (end_text_target in s_text or s_text in end_text_target):
                         final_end_time = transcript_segments[i]['end']
                         break
            
            # Ajusta o fim para um fim de frase da transcricao dentro de [min, max]
            # (antes: "inicio + minimo" quando o end_text nao era achado, e corte seco no maximo)
            desired_end = final_end_time if final_end_time != -1 else None
            final_end_time, snap_reason = snap_end(final_start_time, desired_end, transcript_segments, tempo_minimo, tempo_maximo)
            duration = final_end_time - final_start_time
            if snap_reason != "ok":
                desired_txt = f"{desired_end - final_start_time:.1f}s" if desired_end is not None else "end_text não encontrado"
                print(f"[INFO] Fim ajustado ({snap_reason}): {desired_txt} -> {duration:.1f}s")

            # Construct Final Segment
            processed_segments.append({
                "title": seg.get('title', 'Viral Segment'),
                "start_time": final_start_time,
                "end_time": final_end_time,
                "hook": final_hook(seg, final_start_time, final_end_time, transcript_segments),
                "reasoning": seg.get('reasoning', ''),
                "score": seg.get('score', 0),
                "duration": duration
            })

        except Exception as e:
            print(f"[WARN] Error processing segment {seg}: {e}")
            continue

    # Bonus "mais repetidos": cortes em trechos que o publico reassistiu sobem na lista
    if heatmap:
        for seg in processed_segments:
            heat = heat_average(heatmap, seg["start_time"], seg["end_time"])
            if heat is not None:
                bonus = int(round(HEAT_SCORE_BONUS * heat))
                seg["replay"] = round(heat, 3)
                try:
                    seg["score"] = int(seg.get("score", 0)) + bonus
                except (TypeError, ValueError):
                    seg["score"] = bonus

    # Deduplication
    unique_segments = []
    processed_segments.sort(key=lambda x: int(x.get('score', 0)), reverse=True)
    
    for candidate in processed_segments:
        is_dup = False
        for existing in unique_segments:
            s1, e1 = candidate['start_time'], candidate['end_time']
            # Simple float equality isn't safe, but max/min handles it
            s2, e2 = existing['start_time'], existing['end_time']
            
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            
            if overlap_end > overlap_start:
                intersection = overlap_end - overlap_start
                if intersection > 5: # more than 5 seconds overlap
                    is_dup = True
                    print(f"[DEBUG] Dropping overlap: '{candidate.get('title')}' ({s1:.1f}-{e1:.1f}) overlaps with '{existing.get('title')}' ({s2:.1f}-{e2:.1f}) by {intersection:.1f}s")
                    break
        if not is_dup:
            unique_segments.append(candidate)

    all_segments = unique_segments
    print(f"[DEBUG] Finished processing. {len(all_segments)} segments valid.")

    if output_count and len(all_segments) > output_count:
        print(f"Filtrando os top {output_count} segmentos de {len(all_segments)} candidatos encontrados nos chunks.")
        all_segments = all_segments[:output_count]

    final_result = {"segments": all_segments}
    
    # Validação básica de que temos start_time
    validated_segments = []
    for seg in final_result['segments']:
        if 'start_time' in seg:
             validated_segments.append(seg)
    
    final_result['segments'] = validated_segments
    
    return final_result


VIDEO_TYPES = ("generic", "podcast", "entrevista", "react", "vlog", "gameplay", "aula")

# Instrucao extra enviada a IA conforme o tipo do video ({video_type_instruction} no prompt.txt)
VIDEO_TYPE_INSTRUCTIONS = {
    "generic": "VIDEO TYPE: General video (no specific format).",
    "podcast": (
        "VIDEO TYPE: Podcast / talk show conversation. Prefer segments where ONE complete idea, story, "
        "opinion or argument is delivered from start to finish. Start where the speaker introduces the idea "
        "(not mid-reply), end right after the conclusion or punchline. Avoid stretches dominated by crosstalk, "
        "reading chat messages, sponsor reads or housekeeping."
    ),
    "entrevista": (
        "VIDEO TYPE: Interview. Prefer a clear question followed by a complete, strong answer. The segment may "
        "start with the interviewer's question if it gives context. End when the answer concludes."
    ),
    "react": (
        "VIDEO TYPE: React video (a creator reacting to other content). Prefer moments of strong, genuine "
        "reaction (surprise, laughter, disagreement, hot takes). Include enough of the content being reacted to "
        "for a stranger to understand the reaction. Avoid long silent watching stretches."
    ),
    "vlog": (
        "VIDEO TYPE: Vlog. Prefer moments with a clear mini-story or action, a strong visual or emotional start, "
        "and a payoff. Skip transitions, greetings and logistics talk."
    ),
    "gameplay": (
        "VIDEO TYPE: Gameplay / live stream. Prefer highlights: impressive plays, fails, funny moments, rage or "
        "hype, unexpected events and strong commentary. The segment must make sense without the rest of the stream."
    ),
    "aula": (
        "VIDEO TYPE: Class / lecture / tutorial. Prefer segments that explain ONE concept, tip or step completely "
        "and are useful on their own. Start at the definition or question, end after the explanation or example. "
        "Avoid segments that depend on slides or steps shown earlier."
    ),
}

# --- Capitulos e "mais repetidos" do YouTube ---
HEAT_PEAK_QUANTILE = 0.85   # trechos no top 15% da curva de repeticao = picos
HEAT_SCORE_BONUS = 15       # pontos extras (0-15) para cortes em trechos muito repetidos


def load_youtube_meta(project_folder):
    path = os.path.join(project_folder, "youtube_meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"[WARN] youtube_meta.json invalido: {e}")
        return {}


def heat_peaks(heatmap, quantile=HEAT_PEAK_QUANTILE):
    """Junta os pontos do heatmap acima do quantil em intervalos [(inicio, fim), ...]."""
    pts = [h for h in (heatmap or []) if h.get("value") is not None]
    if len(pts) < 10:
        return []
    vals = sorted(h["value"] for h in pts)
    thr = vals[min(len(vals) - 1, int(len(vals) * quantile))]
    peaks = []
    for h in sorted(pts, key=lambda x: x["start_time"]):
        if h["value"] >= thr:
            if peaks and h["start_time"] <= peaks[-1][1] + 0.5:
                peaks[-1][1] = h["end_time"]
            else:
                peaks.append([h["start_time"], h["end_time"]])
    return [(a, b) for a, b in peaks]


def heat_average(heatmap, start, end):
    """Media da curva de repeticao (0-1) no intervalo, ponderada pela sobreposicao."""
    total = weight = 0.0
    for h in heatmap or []:
        ov = min(end, h["end_time"]) - max(start, h["start_time"])
        if ov > 0 and h.get("value") is not None:
            total += h["value"] * ov
            weight += ov
    return total / weight if weight else None


def split_text(content, chunk_size):
    """Divide texto em pedacos de ate chunk_size caracteres (quebra em espaco, com sobreposicao)."""
    overlap_size = max(1000, int(chunk_size * 0.1))
    chunks, start, n = [], 0, len(content)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            last_space = content.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        if content[start:end].strip():
            chunks.append(content[start:end])
        if end >= n:
            break
        next_start = max(start + 1, end - overlap_size)
        safe_space = content.rfind(' ', start, next_start)
        start = safe_space + 1 if safe_space != -1 else next_start
    return chunks


def chapter_texts(transcript_segments, chapters):
    """Texto da transcricao de cada capitulo, com o titulo no inicio."""
    out = []
    for ch in chapters:
        a, b = float(ch.get("start_time") or 0), float(ch.get("end_time") or 0)
        segs = [x for x in transcript_segments if a <= x["start"] < b]
        if segs:
            out.append(f"[Chapter: {ch.get('title', '')}] " + preprocess_transcript_for_ai(segs))
    return out


def pack_chunks(texts, chunk_size):
    """Agrupa textos (capitulos) em chunks de ate chunk_size, sem cortar capitulo no meio."""
    chunks, cur = [], ""
    for t in texts:
        if len(t) > chunk_size:
            if cur:
                chunks.append(cur); cur = ""
            chunks.extend(split_text(t, chunk_size))
        elif len(cur) + len(t) + 1 > chunk_size:
            chunks.append(cur); cur = t
        else:
            cur = (cur + " " + t).strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_time_range(chunk):
    tags = [int(t) for t in re.findall(r'\((\d+)s\)', chunk)]
    return (min(tags), max(tags)) if tags else (None, None)


def call_provider(ai_mode, prompt, api_key, model_name, base_url=None):
    if ai_mode == "gemini":
        return call_gemini(prompt, api_key, model_name=model_name)
    if ai_mode == "openai":
        return call_openai_compatible(prompt, api_key, model_name=model_name, base_url=base_url)
    if ai_mode == "anthropic":
        return call_anthropic(prompt, api_key, model_name=model_name)
    if ai_mode == "g4f":
        return call_g4f(prompt, model_name=model_name)
    return ""


def select_chapters_by_theme(chapters, themes, ai_mode, api_key, model_name, base_url=None):
    """Chamada rapida so com os titulos: quais capitulos falam dos temas. Retorna indices ou None."""
    listing = "\n".join(f"{i}: {c.get('title', '')}" for i, c in enumerate(chapters))
    prompt = ("These are the chapter titles of a long video:\n" + listing +
              f"\n\nWhich chapters are likely to discuss any of these themes: {themes}?\n"
              "Be inclusive: when unsure, include the chapter. "
              'Answer with JSON only, like {"chapters": [0, 3, 4]}.')
    try:
        text = call_provider(ai_mode, prompt, api_key, model_name, base_url) or ""
        m = re.search(r'"chapters"\s*:\s*\[([^\]]*)\]', text)
        nums = re.findall(r'\d+', m.group(1)) if m else []
        idx = sorted({int(i) for i in nums if 0 <= int(i) < len(chapters)})
        return idx or None
    except AIQuotaError:
        raise
    except Exception as e:
        print(f"[WARN] Filtro de capitulos falhou ({e}); usando todos.")
        return None


def create(num_segments, viral_mode, themes, tempo_minimo, tempo_maximo, ai_mode="manual", api_key=None, project_folder="tmp", chunk_size_arg=None, model_name_arg=None, video_type="generic", base_url=None):
    quantidade_de_virals = num_segments
    for k in AI_STATS:
        AI_STATS[k] = 0
    if video_type not in VIDEO_TYPE_INSTRUCTIONS:
        print(f"[WARN] Tipo de vídeo desconhecido '{video_type}'. Usando 'generic'.")
        video_type = "generic"
    video_type_instruction = VIDEO_TYPE_INSTRUCTIONS[video_type]

    # 1. Load Transcript
    transcript_segments = load_transcript(project_folder)

    # 2. Pre-process Content
    formatted_content = preprocess_transcript_for_ai(transcript_segments)
    content = formatted_content

    # Load Config and Prompt
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'api_config.json')
    prompt_path = os.path.join(base_dir, 'prompt.txt')

    config = {
        "selected_api": "gemini",
        "gemini": {
            "api_key": "",
            "model": "gemini-flash-lite-latest",
            "chunk_size": 200000
        },
        "g4f": {
            "chunk_size": 2000
        },
        "openai": {
            "api_key": "",
            "model": DEFAULT_OPENAI_MODEL,
            "base_url": DEFAULT_OPENAI_BASE_URL,
            "chunk_size": 100000
        },
        "anthropic": {
            "api_key": "",
            "model": DEFAULT_ANTHROPIC_MODEL,
            "chunk_size": 100000
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                for section in ("gemini", "g4f", "openai", "anthropic"):
                    if isinstance(loaded_config.get(section), dict):
                        config[section].update(loaded_config[section])
                if "selected_api" in loaded_config: config["selected_api"] = loaded_config["selected_api"]
        except Exception as e:
            print(f"Erro ao ler api_config.json: {e}")

    # Config Vars
    current_chunk_size = 15000
    model_name = ""
    
    if ai_mode == "gemini":
        cfg_chunk = config["gemini"].get("chunk_size", 200000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        cfg_model = config["gemini"].get("model", "gemini-flash-lite-latest")
        model_name = model_name_arg if model_name_arg else cfg_model
        if not api_key: api_key = config["gemini"].get("api_key", "")
            
    elif ai_mode == "g4f":
        cfg_chunk = config["g4f"].get("chunk_size", 2000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        cfg_model = config["g4f"].get("model")
        model_name = model_name_arg if model_name_arg else cfg_model

    elif ai_mode in ("openai", "anthropic"):
        section = config[ai_mode]
        cfg_chunk = section.get("chunk_size", 100000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        model_name = model_name_arg if model_name_arg else section.get("model")
        env_var = "OPENAI_API_KEY" if ai_mode == "openai" else "ANTHROPIC_API_KEY"
        if not api_key:
            api_key = section.get("api_key", "") or os.environ.get(env_var, "")
        if ai_mode == "openai":
            base_url = base_url or section.get("base_url") or DEFAULT_OPENAI_BASE_URL

    elif ai_mode == "local":
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else 3000
        model_name = model_name_arg if model_name_arg else ""

    system_prompt_template = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_template = f.read()
    else:
        print("Aviso: prompt.txt não encontrado. Usando prompt interno.")
        system_prompt_template = """You are a World-Class Viral Video Editor.
{context_instruction}
{video_type_instruction}
Task: {virality_instruction}
Analyze the transcript below with time tags (XXs). Find {amount} viral segments.
Constraints: Each segment MUST be between {min_duration} seconds and {max_duration} seconds.
IMPORTANT: Output "Title", "Hook", and "Reasoning" in the SAME LANGUAGE as the transcript (e.g., if transcript is Portuguese, output Portuguese).
The "hook" is the short on-screen text (max 10 words) shown at the start; it must be different from the title and describe this specific moment.
TRANSCRIPT:
{transcript_chunk}
OUTPUT JSON ONLY:
{json_template}"""


    json_template = '''
            { "segments" :
                [
                    {
                        "start_text": "Exact first 5-10 words of the segment",
                        "end_text": "Exact last 5-10 words of the segment",
                        "start_time_ref": "Value of closest (XXs) tag",
                        "title": "Catchy clip title, max ~60 characters (Same Language as Transcript)",
                        "hook": "On-screen opening text, max 10 words, DIFFERENT from the title, about THIS moment (Same Language as Transcript)",
                        "reasoning": "One short sentence: why this works (Same Language as Transcript)",
                        "score": 95
                    }
                ]
            }
        '''

    # Chunking: pelos capitulos do YouTube quando existirem; senao, por tamanho fixo
    chunk_size = int(current_chunk_size)
    yt_meta = load_youtube_meta(project_folder)
    chapters = [c for c in (yt_meta.get("chapters") or []) if c.get("end_time")]
    heatmap = yt_meta.get("heatmap") or []
    peaks = heat_peaks(heatmap)

    chunks = []
    if len(chapters) >= 2:
        has_themes = bool((themes or "").strip()) and not viral_mode
        if has_themes and ai_mode in ("gemini", "openai", "anthropic", "g4f"):
            keep = select_chapters_by_theme(chapters, themes, ai_mode, api_key, model_name, base_url)
            if keep:
                print(f"Capitulos sobre os temas: {len(keep)} de {len(chapters)} -> " +
                      "; ".join(chapters[i].get('title', '') for i in keep))
                chapters = [chapters[i] for i in keep]
        texts = chapter_texts(transcript_segments, chapters)
        chunks = pack_chunks(texts, chunk_size)
        print(f"[DEBUG] Chunking por {len(chapters)} capitulo(s) -> {len(chunks)} chunk(s) (limite {chunk_size})")
    if not chunks:
        print(f"[DEBUG] Chunking content (Size: {len(content)}) with Chunk Size: {chunk_size}")
        chunks = split_text(content, chunk_size)
    if peaks:
        print(f"YouTube 'mais repetidos': {len(peaks)} pico(s) marcados para a IA.")
    if ai_mode in ("gemini", "openai", "anthropic", "g4f"):
        print(f"[IA] {len(chunks)} requisição(ões) para escolher os cortes (modelo: {model_name or 'padrão'}).")

    if viral_mode or not (themes or "").strip():
        virality_instruction = f"identify the {quantidade_de_virals} segments with the highest viral potential"
    else:
        virality_instruction = (f"identify the {quantidade_de_virals} best segments about these themes: {themes} "
                                f"(only segments that are clearly about at least one of these themes)")

    output_texts = []
    for i, chunk in enumerate(chunks):
        context_instruction = ""
        if len(chunks) > 1:
            context_instruction = f"Part {i+1} of {len(chunks)}. "
        if peaks:
            lo, hi = chunk_time_range(chunk)
            inside = [(a_, b_) for a_, b_ in peaks if lo is not None and b_ >= lo and a_ <= hi]
            if inside:
                spans = ", ".join(f"{int(a_)}s-{int(b_)}s" for a_, b_ in inside)
                context_instruction += (f"YouTube viewers rewatched these moments the most (strong signal of "
                                        f"interest): {spans}. Prefer segments that include them. ")
        
        try:
            prompt = system_prompt_template.format(
                context_instruction=context_instruction,
                virality_instruction=virality_instruction,
                video_type_instruction=video_type_instruction,
                min_duration=tempo_minimo,
                max_duration=tempo_maximo,
                transcript_chunk=chunk,
                json_template=json_template,
                amount=quantidade_de_virals
            )
        except KeyError as e:
            prompt = system_prompt_template
            prompt = prompt.replace("{context_instruction}", context_instruction)
            prompt = prompt.replace("{virality_instruction}", virality_instruction)
            prompt = prompt.replace("{video_type_instruction}", video_type_instruction)
            prompt = prompt.replace("{min_duration}", str(tempo_minimo))
            prompt = prompt.replace("{max_duration}", str(tempo_maximo))
            prompt = prompt.replace("{transcript_chunk}", chunk)
            prompt = prompt.replace("{json_template}", json_template)
            prompt = prompt.replace("{amount}", str(quantidade_de_virals))

        output_texts.append(prompt)

    try:
        full_prompt_path = os.path.join(project_folder, "prompt_full.txt")
        full_prompt = system_prompt_template
        full_prompt = full_prompt.replace("{context_instruction}", "Full Video Transcript Analysis")
        full_prompt = full_prompt.replace("{virality_instruction}", virality_instruction)
        full_prompt = full_prompt.replace("{video_type_instruction}", video_type_instruction)
        full_prompt = full_prompt.replace("{min_duration}", str(tempo_minimo))
        full_prompt = full_prompt.replace("{max_duration}", str(tempo_maximo))
        full_prompt = full_prompt.replace("{transcript_chunk}", content) 
        full_prompt = full_prompt.replace("{json_template}", json_template)
        full_prompt = full_prompt.replace("{amount}", str(quantidade_de_virals))
        
        with open(full_prompt_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
    except Exception as e:
        print(f"[WARN] Could not save prompt_full.txt: {e}")

    all_raw_segments = []

    print(f"Processando {len(output_texts)} chunks usando modo: {ai_mode.upper()}")

    local_llm_instance = None
    if ai_mode == "local":
        if not HAS_LLAMA_CPP:
            print("Error: llama-cpp-python not installed. Please install it to use Local mode.")
            return {"segments": []}
            
        models_dir = os.path.join(base_dir, 'models')
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
             if os.path.exists(model_name):
                 model_path = model_name
             else:
                 print(f"Error: Model not found at {model_path}")
                 return {"segments": []}
        
        print(f"[INFO] Loading Local Model: {os.path.basename(model_path)} (This may take a while)...")
        try:
            local_llm_instance = Llama(
                model_path=model_path,
                n_gpu_layers=-1, 
                n_ctx=8192,
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            return {"segments": []}

    def _ask(i, prompt):
        response_text = ""
        manual_prompt_path = os.path.join(project_folder, f"prompt_part_{i+1}.txt")
        try:
            with open(manual_prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            print(f"[ERRO] Falha ao salvar prompt.txt: {e}")
        
        if ai_mode == "manual":
            print(f"\n[INFO] O prompt foi salvo em: {manual_prompt_path}")
            print("\n" + "="*60)
            print(f"CHUNK {i+1}/{len(output_texts)}")
            print("="*60)
            print("COPIE O PROMPT ABAIXO (OU DO ARQUIVO GERADO) E COLE NA SUA IA PREFERIDA:")
            print("-" * 20)
            print(prompt)
            print("-" * 20)
            print("="*60)
            print("Cole o JSON de resposta abaixo e pressione ENTER.")
            print("Dica: Se o JSON tiver múltiplas linhas, tente colar tudo de uma vez ou minificado.")
            print("Se preferir, digite 'file' para ler de um arquivo 'tmp/response.json'.")
            
            user_input = input("JSON ou 'file': ")
            
            if user_input.lower() == 'file':
                try:
                    response_json_path = os.path.join(project_folder, 'response.json')
                    with open(response_json_path, 'r', encoding='utf-8') as rf:
                        response_text = rf.read()
                except FileNotFoundError:
                    print(f"Arquivo {response_json_path} não encontrado.")
            else:
                response_text = user_input
                if response_text.strip().startswith("{") and not response_text.strip().endswith("}"):
                    print("Parece incompleto. Cole o resto e dê Enter (ou Ctrl+C para cancelar):")
                    try:
                        rest = sys.stdin.read() 
                        response_text += rest
                    except:
                        pass

        elif ai_mode == "gemini":
            print(f"Enviando chunk {i+1} para o Gemini (Model: {model_name})...")
            response_text = call_gemini(prompt, api_key, model_name=model_name,
                                        response_schema=SEGMENTS_RESPONSE_SCHEMA)
        elif ai_mode == "g4f":
            print(f"Enviando chunk {i+1} para o G4F (Model: {model_name})...")
            response_text = call_g4f(prompt, model_name=model_name)
        elif ai_mode == "openai":
            print(f"Enviando chunk {i+1} para {base_url} (Model: {model_name})...")
            response_text = call_openai_compatible(prompt, api_key, model_name=model_name, base_url=base_url)
        elif ai_mode == "anthropic":
            print(f"Enviando chunk {i+1} para a Anthropic (Model: {model_name})...")
            response_text = call_anthropic(prompt, api_key, model_name=model_name)
        elif ai_mode == "local" and local_llm_instance:
            print(f"Processing chunk {i+1} with Local LLM...")
            try:
                output = local_llm_instance.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    temperature=0.7
                )
                response_text = output['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error evaluating local model: {e}")
                response_text = "{}"

        return response_text

    cloud_modes = ("gemini", "openai", "anthropic", "g4f")
    responses = [None] * len(output_texts)
    if ai_mode in cloud_modes and len(output_texts) > 1:
        workers = max(1, min(int(os.environ.get("VIRALCUTTER_AI_WORKERS", "3")), len(output_texts)))
        print(f"Enviando {len(output_texts)} chunks em paralelo ({workers} por vez)...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ask, i, pr): i for i, pr in enumerate(output_texts)}
            quota_error = None
            for fut in futures:
                i = futures[fut]
                try:
                    responses[i] = fut.result()
                except AIQuotaError as e:
                    quota_error = quota_error or e
                    responses[i] = ""
                except Exception as e:
                    print(f"Erro no chunk {i+1}: {e}")
                    responses[i] = ""
        if quota_error:
            raise quota_error
    else:
        for i, pr in enumerate(output_texts):
            responses[i] = _ask(i, pr)

    for i, response_text in enumerate(responses):
        response_text = response_text or ""
        # --- Save RAW Response for Debugging ---
        try:
            raw_response_path = os.path.join(project_folder, f"response_raw_part_{i+1}.txt")
            with open(raw_response_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"[DEBUG] Raw response saved to: {raw_response_path}")
        except Exception as e:
            print(f"[WARN] Failed to save raw response: {e}")

        # Processar resposta
        try:
            data = clean_json_response(response_text)
            chunk_segments = data.get("segments", [])
            print(f"Encontrados {len(chunk_segments)} segmentos neste chunk.")
            all_raw_segments.extend(chunk_segments)
        except json.JSONDecodeError:
            print(f"Erro: Resposta inválida.")
        except Exception as e:
            print(f"Erro desconhecido ao processar chunk: {e}")

    if AI_STATS["requests"]:
        print(f"[IA] Requisições nesta rodada: {AI_STATS['requests']} | tokens: entrada {AI_STATS['input_tokens']}, "
              f"saída {AI_STATS['output_tokens']}, raciocínio {AI_STATS['thinking_tokens']}", flush=True)

    # Call the alignment / processing logic
    result = process_segments(
        all_raw_segments, 
        transcript_segments, 
        tempo_minimo, 
        tempo_maximo, 
        output_count=quantidade_de_virals,
        heatmap=heatmap
    )
    if not result.get("segments") and ai_mode != "manual":
        print("[WARN] A IA não devolveu nenhum corte aproveitável. Usando a heurística de reserva "
              "('mais repetidos' + densidade de fala). Revise os cortes.", flush=True)
        result = heuristic_segments(transcript_segments, quantidade_de_virals, tempo_minimo, tempo_maximo, heatmap)
    return result
