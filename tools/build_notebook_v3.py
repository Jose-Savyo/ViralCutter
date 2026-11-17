"""Gera o ViralCutter_v3.ipynb (notebook do Colab, modo automático, versão 3.0).

Uso: python tools/build_notebook_v3.py [saida.ipynb]
O notebook tem 4 células: instalar, configurar, processar, ver e baixar.
"""
import json
import os
import sys

REPO_URL = "https://github.com/Jose-Savyo/ViralCutter.git"
REPO_REF = "feat/edicao-humanizada"

INTRO = """# ViralCutter 3.0 — modo automático (sem interface)

Recebe um **link do YouTube** ou um **vídeo seu** e faz tudo sozinho: download, transcrição, escolha dos trechos pela IA, enquadramento 9:16, legendas e ZIP final.

**Antes de começar**
- Menu **Ambiente de execução → Alterar tipo de ambiente de execução → GPU (T4)**.
- Na barra lateral, em 🔑 **Secrets**, crie o secret do provedor de IA (`GEMINI_API_KEY`, `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY`) e ligue o acesso do notebook. Sem secret, a chave é pedida na hora (e não fica salva).

Execute as células na ordem: **1 → 2 → 3 → 4**.

## O que mudou da versão anterior para a 3.0
- **Câmera mais humana:** fica parada enquanto a pessoa se mexe pouco; reenquadra com movimento suave (acelera e freia); ignora falsos positivos da detecção; troca de pessoa vira **corte seco** (como um editor faz), com tempo mínimo em cada plano; sem "degraus" no movimento.
- **Respiro no começo e no fim:** o corte entra um pouco antes da primeira palavra e sai um pouco depois da última, terminando no fim de uma frase; fade curto no áudio e volume padronizado entre os cortes.
- **Legendas por frase:** quebram na pontuação e nas pausas, sem palavra sozinha na tela.
- **Hook diferente do título** (texto curto do momento, até 10 palavras).
- **Mais rápido:** detecção de rosto com só os modelos necessários (2–4× mais rápida), um encode a menos por corte, download em H.264 e cortes processados em paralelo.
- **IA mais robusta:** SDK novo do Gemini, com limite de tempo, raciocínio baixo, resposta em JSON e aviso claro quando a cota diária acaba (em vez de travar).
- **Opcionais por tipo de vídeo:** zoom de ênfase (punch-in) e remoção de pausas longas; remoção de vícios ("é...", "hum") experimental.
- **Tempo de cada etapa no log** e em `timings.json`.

## Como escolher as opções (célula 2)
| Opção | Quando usar |
|---|---|
| **Tipo de vídeo** | `podcast`/`entrevista`: câmera calma, punch-in e remoção de pausas. `react`/`gameplay`: vídeo inteiro no centro com zoom nos rostos. `vlog`: câmera mais rápida. `aula`: vídeo inteiro com fundo desfocado. `generic`: padrão. |
| **Layout** | `pelo tipo` (recomendado), ou force `crop` (9:16 seguindo rostos), `fit-blur` (vídeo inteiro + fundo desfocado), `fit-faces` (vídeo inteiro + zoom nos rostos). |
| **Câmera** | `calm` segura mais o plano; `dynamic` acompanha mais rápido (bom para quem anda). |
| **Provedor/modelo** | `gemini` + `gemini-flash-lite-latest` (padrão: rápido e com cota grande no plano gratuito). `gemini-flash-latest` é melhor, mas o plano gratuito permite só ~20 pedidos por dia. `openai` serve para OpenAI, OpenRouter, Groq, DeepSeek; `anthropic` para Claude. |
| **Cortes em paralelo** | `auto` usa 2 na GPU. Aumente só com GPU maior (L4/A100). |
"""

CELL1 = r'''#@title 1. 🛠️ Instalação (≈5–8 min, só uma vez por sessão)
import os, subprocess, sys, textwrap, glob

#@markdown Repositório e branch do código.
REPO_URL    = "__REPO_URL__" #@param {type:"string"}
REPO_REF    = "__REPO_REF__" #@param {type:"string"}
REPO_DIR    = "/content/ViralCutter"
VENV_PY     = f"{REPO_DIR}/.venv/bin/python"

def run(cmd, **kw):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"Falhou: {cmd}")
    return r.stdout

gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
if not gpu:
    raise SystemExit("❌ Sem GPU. Vá em Ambiente de execução → Alterar tipo → GPU (T4) e rode de novo.")
print("✅ GPU:", gpu)

run("apt-get -qq update && apt-get -qq install -y ffmpeg fonts-montserrat > /dev/null")

if os.path.isdir(REPO_DIR):
    atual = subprocess.run(f"git -C {REPO_DIR} remote get-url origin", shell=True, text=True, capture_output=True).stdout.strip()
    if atual != REPO_URL:
        run(f"rm -rf {REPO_DIR}")
if not os.path.isdir(REPO_DIR):
    run(f"git clone -q {REPO_URL} {REPO_DIR}")
run(f"git -C {REPO_DIR} fetch -q origin")
eh_branch = subprocess.run(f"git -C {REPO_DIR} rev-parse --verify -q origin/{REPO_REF}", shell=True, capture_output=True).returncode == 0
if eh_branch:
    run(f"git -C {REPO_DIR} checkout -q -B {REPO_REF} origin/{REPO_REF}")
else:
    run(f"git -C {REPO_DIR} checkout -q {REPO_REF}")
print("Código:", run(f"git -C {REPO_DIR} log -1 --format='%h %s'").strip())

run("pip install -q uv")
run(f"rm -rf {REPO_DIR}/.venv && uv venv -q --python 3.12 {REPO_DIR}/.venv")

# ctranslate2 4.4 (exigido pelo whisperx 3.3.1) usa cuDNN 8; o torch 2.5.1 traz cuDNN 9.
with open("/content/vc_overrides.txt", "w") as f:
    f.write("ctranslate2==4.5.0\n")

REQS = textwrap.dedent("""
    torch==2.5.1
    torchaudio==2.5.1
    whisperx==3.3.1
    faster-whisper==1.1.0
    pyannote.audio==3.3.2
    transformers==4.39.3
    huggingface-hub<1.0
    numpy<2
    onnxruntime-gpu==1.20.2
    insightface==0.7.3
    mediapipe==0.10.21
    opencv-python==4.11.0.86
    protobuf<5
    yt-dlp
    google-genai
    json-repair
    deep-translator
    ffmpeg-python
    psutil
    tqdm
    matplotlib-inline
    anthropic
    requests
""")
with open("/content/vc_requirements.txt", "w") as f:
    f.write(REQS)

print("⏳ Instalando bibliotecas...")
run(f"uv pip install -q --python {VENV_PY} -r /content/vc_requirements.txt "
    "--override /content/vc_overrides.txt "
    "--extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match")
# faster-whisper puxa o onnxruntime de CPU, que sobrescreve o de GPU: troca de volta sem mexer em numpy/protobuf.
run(f"uv pip uninstall -q --python {VENV_PY} onnxruntime")
run(f"uv pip install -q --python {VENV_PY} --reinstall-package onnxruntime-gpu onnxruntime-gpu==1.20.2 "
    "-c /content/vc_requirements.txt")

check = textwrap.dedent("""
    import warnings; warnings.filterwarnings("ignore")
    import numpy, google.protobuf, ctranslate2
    import torch, onnxruntime as ort, whisperx, insightface, mediapipe as mp, cv2
    from google import genai
    import json_repair
    from faster_whisper import WhisperModel
    from faster_whisper.utils import _MODELS
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| cuDNN", torch.backends.cudnn.version())
    print("ctranslate2", ctranslate2.__version__, "| onnxruntime providers:", ort.get_available_providers())
    print("numpy", numpy.__version__, "| protobuf", google.protobuf.__version__,
          "| mediapipe", mp.__version__, "| opencv", cv2.__version__, "| google-genai", genai.__version__)
    assert torch.cuda.is_available(), "torch sem CUDA"
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "onnxruntime sem CUDA"
    assert numpy.__version__.startswith("1."), "numpy 2 instalado (quebra o mediapipe)"
    assert int(google.protobuf.__version__.split(".")[0]) < 5, "protobuf >= 5 instalado"
    assert hasattr(mp, "solutions"), "mediapipe sem mp.solutions"
    assert "large-v3-turbo" in _MODELS, "faster-whisper sem large-v3-turbo"
    m = WhisperModel("tiny", device="cuda", compute_type="float16")
    segs, _ = m.transcribe(numpy.zeros(16000 * 2, dtype=numpy.float32))
    list(segs)
    print("teste de transcrição na GPU: OK")
""")
with open("/content/vc_check.py", "w") as f:
    f.write(check)
# MPLBACKEND=Agg: o Colab exporta um backend que não existe no venv.
# LD_LIBRARY_PATH: o ctranslate2 só acha o cuDNN 9 do venv por aqui.
nvidia_libs = glob.glob(f"{REPO_DIR}/.venv/lib/python3.12/site-packages/nvidia/*/lib")
assert any("cudnn" in p for p in nvidia_libs), "cuDNN não encontrado no venv"
check_env = dict(os.environ, MPLBACKEND="Agg",
                 LD_LIBRARY_PATH=":".join(nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")]))
print(run(f"{VENV_PY} /content/vc_check.py", cwd=REPO_DIR, env=check_env))
print("✅ Instalação concluída.")
'''

CELL2 = r'''#@title 2. ⚙️ Configuração
#@markdown ### Entrada
fonte = "YouTube" #@param ["YouTube", "Upload do computador", "Google Drive"]
youtube_url = "" #@param {type:"string"}
#@markdown Caminho do vídeo no Drive (só para a opção Google Drive), ex.: `/content/drive/MyDrive/videos/aula.mp4`
caminho_drive = "" #@param {type:"string"}
qualidade_download = "1080p" #@param ["best", "1080p", "720p", "480p"]
#@markdown Se o YouTube bloquear o download, envie um `cookies.txt` (formato Netscape) e informe o caminho:
cookies_txt = "" #@param {type:"string"}

#@markdown ### Cortes
quantidade_cortes = 3 #@param {type:"slider", min:1, max:15, step:1}
duracao_minima = 30 #@param {type:"integer"}
duracao_maxima = 60 #@param {type:"integer"}
#@markdown Deixe `temas` vazio para o modo viral automático, ou informe temas separados por vírgula.
temas = "" #@param {type:"string"}

#@markdown ### IA que escolhe os cortes
#@markdown `openai` = qualquer API compatível com OpenAI (OpenAI, OpenRouter, Groq, DeepSeek...). Chave no secret `GEMINI_API_KEY`, `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY`.
provedor_ia = "gemini" #@param ["gemini", "openai", "anthropic"]
modelo_ia = "gemini-flash-lite-latest" #@param ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o-mini", "claude-opus-5-5", "claude-sonnet-5-5"] {allow-input: true}
#@markdown Só para `openai`: vazio = OpenAI. OpenRouter: `https://openrouter.ai/api/v1` · Groq: `https://api.groq.com/openai/v1` · DeepSeek: `https://api.deepseek.com/v1`
url_base_api = "" #@param {type:"string"}

#@markdown ### Tipo de vídeo, enquadramento e câmera
#@markdown O tipo define os padrões de layout, câmera, punch-in e remoção de pausas, e orienta a IA.
tipo_video = "generic" #@param ["generic", "podcast", "entrevista", "react", "vlog", "gameplay", "aula"]
layout = "pelo tipo" #@param ["pelo tipo", "crop", "fit-blur", "fit-faces"]
#@markdown `calm` segura mais o plano · `normal` · `dynamic` acompanha mais rápido
camera = "pelo tipo" #@param ["pelo tipo", "calm", "normal", "dynamic"]
modo_rosto = "pelo tipo" #@param ["pelo tipo", "auto", "1", "2"]
#@markdown Quando nenhum rosto é encontrado: `padding` (barras) ou `zoom` (recorte central)
sem_rosto = "padding" #@param ["padding", "zoom"]
focar_quem_fala = False #@param {type:"boolean"}

#@markdown ### Edição
#@markdown Zoom de ênfase (punch-in) nos picos da fala, no layout `crop`:
punch_in = "pelo tipo" #@param ["pelo tipo", "sim", "não"]
#@markdown Encurtar pausas longas (só quando o trecho está silencioso e parado):
remover_pausas = "pelo tipo" #@param ["pelo tipo", "sim", "não"]
#@markdown Experimental: remover hesitações ("é...", "hum", "ahn"):
remover_vicios = False #@param {type:"boolean"}

#@markdown ### Transcrição
modelo_whisper = "large-v3-turbo" #@param ["large-v3-turbo", "large-v3", "medium", "small"]
#@markdown Usar as legendas do YouTube em vez do Whisper: pula transcrição e alinhamento (bem mais rápido; a legenda automática tem menos pontuação)
usar_legendas_youtube = False #@param {type:"boolean"}
#@markdown Batch do Whisper: `0` = automático pela VRAM (T4 → 32, L4 → 48, A100 → 64).
batch_whisper = 0 #@param {type:"integer"}

#@markdown ### Desempenho
#@markdown Quantos cortes enquadrar ao mesmo tempo (`auto` = 2 na GPU).
cortes_em_paralelo = "auto" #@param ["auto", "1", "2", "3", "4"]

#@markdown ### Legendas
#@markdown Código do idioma para traduzir as legendas (ex.: `en`, `es`). Vazio = sem tradução.
traduzir_para = "" #@param {type:"string"}

#@markdown ### Saída
salvar_no_drive = False #@param {type:"boolean"}
pasta_drive_saida = "/content/drive/MyDrive/ViralCutter" #@param {type:"string"}

if duracao_minima >= duracao_maxima:
    raise ValueError("duracao_minima precisa ser menor que duracao_maxima")
if fonte == "YouTube" and not youtube_url.strip():
    raise ValueError("Informe o youtube_url")
_prefixos = {"gemini": "gemini", "anthropic": "claude"}
if provedor_ia in _prefixos and not modelo_ia.startswith(_prefixos[provedor_ia]):
    print(f"⚠️ O modelo '{modelo_ia}' não parece ser do provedor '{provedor_ia}'. Confira modelo_ia.")
if provedor_ia == "gemini" and modelo_ia == "gemini-flash-latest":
    print("ℹ️ gemini-flash-latest: o plano gratuito permite ~20 pedidos por dia. Se a cota acabar, use gemini-flash-lite-latest.")
print("✅ Configuração salva. Rode a célula 3.")
'''

CELL3 = r'''#@title 3. 🚀 Executar tudo
import os, re, sys, json, shutil, subprocess, glob, time, getpass

REPO_DIR = "/content/ViralCutter"
VENV_PY  = f"{REPO_DIR}/.venv/bin/python"
assert os.path.exists(VENV_PY), "Rode a célula 1 (Instalação) primeiro."

# --- Chave da IA (Secrets do Colab ou digitada; nunca gravada) ---
SECRET_POR_PROVEDOR = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
secret_name = SECRET_POR_PROVEDOR[provedor_ia]
api_key = None
try:
    from google.colab import userdata
    api_key = userdata.get(secret_name)
except Exception:
    pass
if not api_key:
    api_key = getpass.getpass(f"Chave para {provedor_ia} ({secret_name}): ").strip()
if not api_key:
    raise SystemExit(f"❌ Sem chave para {provedor_ia}. Crie o secret {secret_name}.")

if fonte == "Google Drive" or salvar_no_drive:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")

def nova_pasta_projeto(nome_arquivo):
    base = os.path.splitext(os.path.basename(nome_arquivo))[0]
    base = re.sub(r"[^\w\- ]", "", base).strip()[:60] or "video"
    pasta = os.path.join(REPO_DIR, "VIRALS", f"{base}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(pasta, exist_ok=True)
    return pasta

project_path = None
origem = None
if fonte == "Upload do computador":
    from google.colab import files
    print("Selecione o vídeo:")
    up = files.upload()
    if not up:
        raise SystemExit("❌ Nenhum arquivo enviado.")
    origem = os.path.abspath(next(iter(up)))
elif fonte == "Google Drive":
    if not os.path.isfile(caminho_drive):
        raise SystemExit(f"❌ Arquivo não encontrado: {caminho_drive}")
    origem = caminho_drive

if origem:
    project_path = nova_pasta_projeto(origem)
    destino = os.path.join(project_path, "input.mp4")
    if origem.lower().endswith(".mp4"):
        print("Preparando vídeo...")
        (shutil.move if fonte == "Upload do computador" else shutil.copy)(origem, destino)
    else:
        print("Convertendo para MP4...")
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", origem, "-c", "copy", destino])
        if r.returncode != 0 or not os.path.exists(destino) or os.path.getsize(destino) == 0:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", origem,
                            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", destino], check=True)
        if fonte == "Upload do computador":
            os.remove(origem)

SIM_NAO = {"sim": "on", "não": "off"}
cmd = [VENV_PY, "-u", "main_improved.py",
       "--skip-prompts", "--workflow", "1",
       "--segments", str(quantidade_cortes),
       "--min-duration", str(duracao_minima), "--max-duration", str(duracao_maxima),
       "--model", modelo_whisper,
       "--ai-backend", provedor_ia, "--api-key", api_key, "--ai-model-name", modelo_ia,
       "--video-type", tipo_video,
       "--face-model", "insightface", "--no-face-mode", sem_rosto,
       "--parallel-clips", cortes_em_paralelo]
if batch_whisper > 0:
    cmd += ["--whisper-batch-size", str(batch_whisper)]
if layout != "pelo tipo":
    cmd += ["--frame-layout", layout]
if camera != "pelo tipo":
    cmd += ["--camera-style", camera]
if modo_rosto != "pelo tipo":
    cmd += ["--face-mode", modo_rosto]
if punch_in != "pelo tipo":
    cmd += ["--punch-in", SIM_NAO[punch_in]]
if remover_pausas != "pelo tipo":
    cmd += ["--remove-pauses", SIM_NAO[remover_pausas]]
if remover_vicios:
    cmd.append("--remove-fillers")
if provedor_ia == "openai" and url_base_api.strip():
    cmd += ["--api-base-url", url_base_api.strip()]
cmd += ["--themes", temas] if temas.strip() else ["--viral"]
if focar_quem_fala:
    cmd.append("--focus-active-speaker")
if traduzir_para.strip():
    cmd += ["--translate-target", traduzir_para.strip()]
if project_path:
    cmd += ["--project-path", project_path]
else:
    cmd += ["--url", youtube_url.strip(), "--video-quality", qualidade_download]
    if not usar_legendas_youtube:
        cmd.append("--skip-youtube-subs")  # força a transcrição com WhisperX
    if cookies_txt.strip():
        cmd += ["--cookies-file", cookies_txt.strip()]

env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
cudnn = glob.glob(f"{REPO_DIR}/.venv/lib/python3.12/site-packages/nvidia/*/lib")
env["LD_LIBRARY_PATH"] = ":".join(cudnn + [env.get("LD_LIBRARY_PATH", "")])

# Linhas que só fazem barulho na tela (continuam no log completo)
RUIDO = re.compile(r"\d+%\|| it/s|s/it|UserWarning|FutureWarning|^\s*warnings\.warn|^DEBUG: Frame |"
                   r"^\[DEBUG\] (Raw response|Matching|Dropping overlap)|^frame=\s*\d+")

log_path = "/content/viralcutter_log.txt"
print("🚀 Processando... (o log completo fica em", log_path, ")\n")
inicio = time.time()
pasta_saida = project_path
with open(log_path, "w", encoding="utf-8") as log:
    p = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    for linha in p.stdout:
        if api_key:
            linha = linha.replace(api_key, "***")
        log.write(linha)
        m = re.search(r"Project Folder: (.+)", linha)
        if m:
            pasta_saida = m.group(1).strip()
        if RUIDO.search(linha):
            continue
        print(linha, end="")
    p.wait()

minutos = (time.time() - inicio) / 60
if p.returncode != 0:
    dica = ""
    if p.returncode == 2:
        dica = "\n➡️ Cota da IA esgotada: troque modelo_ia (ex.: gemini-flash-lite-latest) ou o provedor na célula 2."
    print(f"\n❌ Falhou após {minutos:.1f} min (código {p.returncode}). Veja o fim do log acima ou {log_path}.{dica}")
else:
    PROJECT_FOLDER = os.path.abspath(os.path.join(REPO_DIR, pasta_saida)) if pasta_saida else None
    tempos = os.path.join(PROJECT_FOLDER or "", "timings.json")
    if PROJECT_FOLDER and os.path.exists(tempos):
        dados = json.load(open(tempos, encoding="utf-8"))
        print("\n⏱️ Tempo por etapa:")
        for etapa in dados.get("stages", []):
            print(f"   {etapa['name']:<14} {etapa['seconds']:>8.1f} s")
        print(f"   {'total':<14} {dados.get('total_seconds', 0):>8.1f} s")
    print(f"\n✅ Concluído em {minutos:.1f} min. Pasta: {PROJECT_FOLDER}\nRode a célula 4 para ver e baixar.")
'''

CELL4 = r'''#@title 4. 🎬 Ver e baixar resultados
import os, glob, shutil, base64
from IPython.display import display, HTML

assert "PROJECT_FOLDER" in globals() and PROJECT_FOLDER, "Rode a célula 3 até o fim primeiro."
videos = sorted(glob.glob(os.path.join(PROJECT_FOLDER, "burned_sub", "*.mp4")))
if not videos:
    videos = sorted(glob.glob(os.path.join(PROJECT_FOLDER, "final", "*.mp4")))
if not videos:
    raise SystemExit(f"Nenhum vídeo encontrado em {PROJECT_FOLDER}")

print(f"{len(videos)} corte(s):")
for v in videos:
    print(f"  • {os.path.basename(v)}  ({os.path.getsize(v)/1e6:.1f} MB)")

for v in videos:
    if os.path.getsize(v) < 25e6:
        b64 = base64.b64encode(open(v, "rb").read()).decode()
        display(HTML(f'<p><b>{os.path.basename(v)}</b></p>'
                     f'<video width="270" controls src="data:video/mp4;base64,{b64}"></video>'))

zip_base = f"/content/{os.path.basename(PROJECT_FOLDER)}_cortes"
pacote = "/content/_pacote"
shutil.rmtree(pacote, ignore_errors=True); os.makedirs(pacote)
for v in videos:
    shutil.copy(v, pacote)
for extra in ("viral_segments.txt", "process_config.json", "timings.json", "timings_clips.json"):
    f = os.path.join(PROJECT_FOLDER, extra)
    if os.path.exists(f):
        shutil.copy(f, pacote)
zip_path = shutil.make_archive(zip_base, "zip", pacote)
print("ZIP:", zip_path)

if salvar_no_drive:
    os.makedirs(pasta_drive_saida, exist_ok=True)
    shutil.copy(zip_path, pasta_drive_saida)
    print("Copiado para o Drive:", pasta_drive_saida)

from google.colab import files
files.download(zip_path)
'''


def _lines(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"},
            "outputs": [], "source": _lines(src)}


def build():
    cell1 = CELL1.replace("__REPO_URL__", REPO_URL).replace("__REPO_REF__", REPO_REF)
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": "ViralCutter_v3.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": _lines(INTRO)},
            code_cell(cell1),
            code_cell(CELL2),
            code_cell(CELL3),
            code_cell(CELL4),
        ],
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ViralCutter_v3.ipynb")
    nb = build()
    for cell in nb["cells"][1:]:
        compile("".join(cell["source"]), "<cell>", "exec")  # erro de sintaxe falha aqui, nao no Colab
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"ok: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
