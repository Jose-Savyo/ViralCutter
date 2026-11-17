# ViralCutter 3.0: pesquisa comparativa e plano

Data: 16/11/2026. Base: branch `feat/layouts-ia-tipo-video` (commit `ae58f26`).

Repositórios lidos (clone raso, código lido, não só o README):

| Repositório | Commit | Licença | Forma |
|---|---|---|---|
| samuelbovef/auto-clipper | d19b2b3 | MIT | scripts Python (Windows .bat) |
| lucianodiisouza/PrimeClip | 40826a4 | MIT | app desktop (Tauri + FastAPI) |
| mutonby/openshorts | a75d8a2 | MIT (mas usa `ultralytics` = AGPL-3.0) | Docker + FastAPI + SaaS |
| FujiwaraChoki/supoclip | 2232b76 | **AGPL-3.0** | Docker + Postgres + Redis |
| RafaelGodoyEbert/ViralCutter (main) | dbe3215 | GPL-3.0 | **igual à nossa base** (nada mudou desde então) |
| marcoaureliojf/viralcutterweb | 12aefad | **sem licença** (todos os direitos reservados) | Docker + FastAPI |

Sobre licenças: o ViralCutter é GPL-3.0. Código MIT pode ser incorporado (mantendo o aviso de copyright).
Do supoclip (AGPL) e do viralcutterweb (sem licença) eu **não copio código**: só reimplemento ideias e
fórmulas públicas (por exemplo, a mola criticamente amortecida, que é uma fórmula conhecida, a SmoothDamp).

---

## Medições feitas na nossa branch (CPU local, antes de qualquer mudança)

| Medição | Resultado |
|---|---|
| InsightFace `buffalo_l` com os 5 modelos (o que roda hoje) | **1.867 ms** por detecção (quadro de transmissão com 6 rostos) |
| Só detecção + landmark 3D (necessário para "quem fala") | **814 ms** (−56%) |
| Só detecção | **285 ms** (−85%) |
| Decodificação OpenCV, 1080p | H.264 145 fps · VP9 119 fps · **AV1: não decodifica** ("doesn't support hardware accelerated AV1") |
| Re-encodes por corte | **4**: corte → enquadramento (pipe) → mux (re-encoda o vídeo em vez de copiar) → legenda |
| Formato baixado | `bestvideo[height<=1080]` → VP9 local, **AV1 (f399) no Colab**. Só funciona porque o corte re-encoda para H.264 antes do OpenCV |
| Gemini `gemini-flash-latest` (plano gratuito) | 20 requisições/dia; ao estourar, o código não percebe que o limite é diário e fica tentando de novo |
| Gemini `gemini-flash-lite-latest` | 7 s no total (16k tokens de entrada, 1,7k de saída) |

---

## 1. Escolha dos cortes

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | Sem IA: densidade de "power words" + bônus para "?" em janelas deslizantes; ou picos de volume (RMS) expandidos até o silêncio mais próximo | `src/cutter.py:88-232`, `src/analisador_emocoes.py:40-160` | sim | MIT |
| PrimeClip | IA recebe transcrição **+ lista de cenas**, devolve início/fim em segundos (JSON); nota 0-10; se a IA falhar, heurística por janelas dentro das cenas | `providers/ollama.py:41-90`, `providers/json_utils.py:23-120`, `prompts/clip_selection.txt` | lógica sim (app é desktop) | MIT |
| openshorts | **Duas passadas**: nota para **todas** as janelas de ~90 s (ranking global), depois "passada de detalhe" só no top N para limites exatos e hook. Quantidade de cortes derivada da duração, com piso. Saída validada por esquema (`response_schema`) | `clip_selection.py`, `main.get_viral_clips`, `gemini_worker.py:519-580`, `docs/architecture/clip-selection.md` | sim (precisa chave Gemini) | MIT |
| supoclip | Transcrição em linhas `[MM:SS - MM:SS]`; IA devolve tempos que são **encaixados no limite de linha mais próximo (até 8 s)** em vez de descartar o corte; rubrica 4×25 (gancho, engajamento, valor, compartilhamento) + tipo de gancho; `hook_title` separado (≤ 64 caracteres / 10 palavras); regras de "grounding" (não inventar, um trecho contínuo) | `backend/src/ai.py:50-345` | parcial (pydantic-ai; AssemblyAI pago é opcional) | AGPL |
| ViralCutter main | igual à nossa base | — | sim | GPL |
| viralcutterweb | Chunks por duração com sobreposição, g4f `gpt_4` | `scripts/create_viral_segments.py` | sim | sem licença |
| **Nossa branch** | IA com `start_text`/`end_text`/`start_time_ref`, chunks em paralelo, filtro por capítulos, bônus de "mais repetidos", instrução por tipo de vídeo, fim encaixado no fim da frase | `scripts/create_viral_segments.py` | sim | GPL |

**Vencedor:** openshorts no método (ranking global + saída com esquema) e supoclip na rubrica e no encaixe de limites.
Capítulos e "mais repetidos" **só a nossa versão tem**.

**Adotar:**
- saída JSON com esquema (o modelo não consegue devolver formato errado) + `json-repair` como rede de segurança (ganho: menos falhas e reenvios; esforço baixo);
- `hook` separado do título, até 10 palavras, com a regra "sobre este momento, não sobre o vídeo" (corrige hook == title; baixo);
- rubrica de 4 notas no prompt, para a nota ficar comparável entre partes (médio);
- heurística de reserva quando a IA não devolver nada: janelas com mais "mais repetidos" ou mais energia de áudio (médio).

## 2. Enquadramento e câmera

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | Só recorte central fixo | `src/cutter.py:242` | sim | MIT |
| PrimeClip | MediaPipe a 6 fps → média móvel + velocidade máxima → **um único recorte parado por corte (mediana)** | `services/vertical_crop/dynamic_crop.py`, `pipelines/clip_pipeline.py:200-230` | sim | MIT |
| openshorts | **"Heavy Tripod"**: zona segura de 25% da largura do recorte; salto grande só é aceito depois de **3 detecções seguidas** (mediram −69% de reversões); na troca de cena **corta seco** para o novo rosto; panorâmica lenta e linear. **SpeakerTracker**: troca de pessoa só depois de confirmar (15 quadros) e com tempo mínimo entre trocas (30 quadros), zerado na troca de cena. Troca de quem fala como **corte seco**, com atividade da boca normalizada por pessoa. Split só se os dois rostos aparecem juntos em ≥ 50% das amostras. Detector de webcam no canto (pequena, fora do centro, parada) | `main.py:116-340`, `active_speaker.py`, `split_layout.py`, `camera_inset.py`, `reframe_v2.py`, `docs/architecture/layouts-and-reframing.md` | algoritmos sim; app exige Docker e Gemini para escolher layout | MIT (+ YOLO AGPL) |
| supoclip | **Mola criticamente amortecida** (SmoothDamp) indo para um alvo com zona de conforto de 5%, filtro de mediana contra picos, passa-baixa final; **recorte parado se o movimento total < 4% da largura**; composição por plano em entrevistas (dupla empilhada / um só / quadro inteiro), trocando só no corte do editor; layout de react exige rosto estável no canto + borda da janela detectável | `media/reframing.py:900-1050`, `media/interview_layout.py`, `media/reaction_layout.py`, `media/talking_head_layout.py` | algoritmos sim | AGPL (só ideias) |
| viralcutterweb | Usuário marca duas regiões na interface web | `scripts/edit_video.py:110-150` | não (manual) | sem licença |
| **Nossa branch** | InsightFace a cada 0,17 s (0,5 s com 2 rostos); zona parada de 11% da altura; transição suave de 0,5 s; corte seco na troca de cena; fit-faces com troca automática para crop e filtro de rosto do conteúdo; recorte em pixel inteiro; decisão 1↔2 rostos a cada detecção | `scripts/edit_video.py`, `scripts/one_face.py` | sim | GPL |

**Vencedor:** openshorts no comportamento (confirmação de salto, histerese de quem fala, corte seco) e supoclip
na curva de movimento (mola) e na regra de "câmera parada quando o movimento é pequeno".

**Adotar (câmera virtual v2):** zona de conforto + confirmação de salto (3 detecções) + mola criticamente amortecida
no lugar da transição fixa de 0,5 s + recorte com subpixel (sem "degraus" no zoom de 1,8×) + travar o plano quando
o movimento for pequeno + tempo mínimo de 1,5 s em cada pessoa/layout, trocando de preferência no corte de cena.
Ganho: o maior impacto em "editado por uma pessoa". Esforço: médio.

## 3. Transições e ritmo

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | Limites do corte nas pausas em volta do pico de volume; zoom lento (`zoompan`) como transformação; supercut | `src/analisador_emocoes.py`, `src/transformador.py:159-175`, `src/montador_reels.py` | sim | MIT |
| PrimeClip | Planejador de zoom por palavra-chave existe, mas **não é aplicado na renderização** (só no README) | `services/motion/planner.py` | — | MIT |
| openshorts | **Punch-in** de 1,12× nos picos do áudio: entra em 0,25 s, segura 1,3 s, sai em 0,55 s, intervalo mínimo de 18 s (≈ 3 por minuto, calibrado em 20 cortes) | `punch_in.py` | sim | MIT |
| supoclip | Zoom lento com superamostragem 1,5× (sem degraus); **remove pausa só se o trecho também estiver parado e silencioso** (não corta ação de gameplay) | `media/reframing.py:1154-1175`, `media/gap_activity.py` | sim | AGPL (só ideias) |
| viralcutterweb | — | — | — | — |
| **Nossa branch** | Nada: corte seco, sem fade, sem zoom, sem remoção de pausa | — | — | — |

**Vencedor:** openshorts (punch-in calibrado) + supoclip (pausa só onde é seguro remover).

**Adotar:** respiro no início e no fim com fade de áudio (baixo, alto impacto); punch-in opcional por tipo (médio);
compressão de pausas opcional, com a checagem de "parado e silencioso" (médio-alto); remoção de vícios como
experimental, desligada por padrão (alto risco de corte no meio de palavra).

## 4. Legendas

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | ASS com animações (fade, slide, pop, flip); N palavras fixas por bloco | `src/ajustar_legendas.py:73-189` | sim | MIT |
| PrimeClip | N palavras por tela; ASS; editor | `services/captions/chunker.py`, `ass_builder.py` | sim | MIT |
| openshorts | Blocos por **limite de caracteres** (9–20 conforme o estilo, calculado pelo tamanho da fonte) e no máximo 2 s; **cada palavra fica até a próxima começar** (sem piscar); no split, legenda na emenda; texto do hook posicionado onde não cobre rosto | `subtitles.py:131-520`, `hook_placement.py` | sim | MIT |
| supoclip | Largura medida com a métrica real da fonte; quebra em saltos > 2 s; posição segura fora das faixas | `media/captions.py` | sim | AGPL (só ideias) |
| viralcutterweb | pycaps (estilos em CSS renderizados num navegador) | `scripts/pycaps_processing.py` | pesado (Chromium) | sem licença |
| **Nossa branch** | ASS com destaque palavra a palavra, 3 palavras por bloco, quebra em pausa > 0,5 s | `scripts/adjust_subtitles.py` | sim | GPL |

**Vencedor:** openshorts nas regras de quebra; o nosso destaque palavra a palavra já é bom.

**Adotar:** quebrar na pontuação e em pausas, limite de caracteres por bloco, nunca deixar uma palavra sozinha
(junta com o bloco anterior), manter o bloco até a próxima fala (baixo-médio).

## 5. Áudio

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | nada | — | — | MIT |
| PrimeClip | `afade` de entrada e saída no editor de linha do tempo | `services/captions/timeline_renderer.py:162-165` | sim | MIT |
| openshorts | `loudnorm=I=-14:TP=-2.0:LRA=11` (TP −2,0 medido: −1,5 ainda estourava depois do AAC) | `ffmpeg_utils.py:80-93` | sim | MIT |
| supoclip | `loudnorm=I=-14:TP=-1.5:LRA=11` | `media/common.py:153`, `media/ffmpeg.py:367` | sim | AGPL |
| viralcutterweb / ViralCutter main | nada | — | — | — |
| **Nossa branch** | nada | — | — | — |

**Vencedor:** openshorts (valores medidos). **Adotar:** loudnorm −14 LUFS / TP −2 + fade de 80 ms na entrada e
250 ms na saída, no mesmo encode da legenda (baixo; volume igual entre cortes e sem "clique" no começo).

## 6. Velocidade

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | Um re-encode libx264 por transformação (lento) | `src/transformador.py` | sim | MIT |
| PrimeClip | **Um único ffmpeg por corte** (busca + recorte + escala + legenda); rostos amostrados a 6 fps | `services/render/ffmpeg_renderer.py:36-75` | sim | MIT |
| openshorts | Decodificação na GPU (NVDEC) com volta para CPU; leitura em passada única em vez de buscas (89 → 6 s de CPU num corte de 20 s); detecção a cada 4 quadros; recorte por quadro via `sendcmd` do ffmpeg (sem passar pixels pelo Python); render próprio em CUDA | `gpu_decode.py`, `frame_sampler.py`, `reframe_v2.py:81-920`, `gpu_render.py` | parcial (o render CUDA não é portável) | MIT |
| supoclip | Movimento como expressão do `crop` do ffmpeg (sem pipe de pixels); análise em baixa resolução (64×36 a 4 fps) | `media/reframing.py:1024-1050`, `media/gap_activity.py` | sim | AGPL |
| **Nossa branch** | Composição quadro a quadro em Python + pipe de vídeo cru; InsightFace com 5 modelos; 4 encodes por corte; cortes processados um por vez; download em AV1 | `scripts/edit_video.py`, `scripts/cut_segments.py`, `scripts/burn_subtitles.py` | sim | GPL |

**Vencedor:** openshorts.

**Adotar:** carregar só os modelos usados do InsightFace (−56% a −85% medido; baixo); mux copiando o vídeo
(−1 encode; baixo); preferir H.264 no download (decodifica no OpenCV e na NVDEC da T4; baixo); cortes em paralelo
(médio); tempo por etapa no log (baixo). Render por expressões do ffmpeg em vez do pipe Python: alto esforço,
fica para depois de medir.

## 7. Robustez

| Repositório | Como faz | Arquivo(s) | Colab grátis? | Licença |
|---|---|---|---|---|
| auto-clipper | try/except com mensagens; detecta áudio mudo | `src/analisador_emocoes.py` | sim | MIT |
| PrimeClip | Timeout de 120 s, 3 tentativas, `json-repair`, heurística quando a IA falha | `providers/ollama.py:74-90`, `providers/json_utils.py` | sim | MIT |
| openshorts | Volta para a CPU quando a GPU falha; caminho por imagem quando o vídeo não tem fala; timeouts nos subprocessos | `gpu_decode.py`, `main.get_visual_clips` | sim | MIT |
| supoclip | Retenta só códigos transitórios (408, 429, 500, 502, 503, 504, 529); timeout total de 30 min e de 4 min no re-ranking; encaixa limites em vez de descartar | `backend/src/ai.py:50-60` | sim | AGPL |
| **Nossa branch** | SDK do Gemini descontinuado, sem timeout; 429 diário tratado como por minuto; fim do corte encaixado | `scripts/create_viral_segments.py` | sim | GPL |

**Vencedor:** supoclip + PrimeClip. **Adotar:** `google-genai` com timeout, distinção entre cota diária
(falha na hora com mensagem e alternativas) e por minuto (espera o tempo pedido, em qualquer formato),
retentativa só nos códigos transitórios, `json-repair`, heurística de reserva (médio).

## 8. Outras ideias

| Ideia | Onde | Vale para nós? |
|---|---|---|
| Detectar legenda já queimada no vídeo original e cortar essa faixa (evita legenda dupla) | openshorts `burned_subtitles.py` | sim, mas usa Gemini com imagens (gasta cota); depois |
| Texto do hook posicionado onde não cobre rosto | openshorts `hook_placement.py` | sim, quando houver hook na tela |
| Escolha de layout pela IA com 12 quadros (~3k tokens) | openshorts `layout_picker.py` | não agora: gasta cota do plano gratuito |
| LUT de cor, supercut, lote de URLs | auto-clipper | fora do escopo |
| Ajuste manual de regiões | viralcutterweb | não (fluxo é automático) |

---

## Lista única das ideias a adotar (por impacto)

| # | Ideia | Impacto | Esforço |
|---|---|---|---|
| 1 | Câmera virtual v2 (zona de conforto + confirmação de salto + mola + subpixel + travar plano + tempo mínimo por pessoa/layout) | humanização ★★★ | médio |
| 2 | Respiro no início/fim + fade de áudio + loudnorm | humanização ★★★ | baixo |
| 3 | InsightFace só com os modelos usados | velocidade ★★★ (−56% a −85%) | baixo |
| 4 | IA: `google-genai`, timeout, cota diária/minuto, JSON com esquema, raciocínio limitado, flash-lite padrão | velocidade e robustez ★★★ | médio |
| 5 | Hook diferente do título | humanização ★★ | baixo |
| 6 | Legendas por frase, sem palavra solta | humanização ★★ | baixo-médio |
| 7 | Cortes em paralelo | velocidade ★★ | médio |
| 8 | Mux sem re-encode + H.264 no download | velocidade ★★ | baixo |
| 9 | Punch-in opcional por tipo | humanização ★★ | médio |
| 10 | Compressão de pausas segura (parado e silencioso), opcional | humanização ★ | médio-alto |
| 11 | Heurística de reserva quando a IA falha | robustez ★★ | médio |
| 12 | Remoção de vícios ("é...", "tipo"), experimental | humanização ★ (risco alto) | alto |
| — | Render por expressões do ffmpeg / CUDA | velocidade | alto (depois) |

---

## Plano (fase 3)

Nova branch `feat/edicao-humanizada` a partir de `feat/layouts-ia-tipo-video`. Um commit por tema, cada um com
teste em clipes de 10–20 s e o antes/depois das métricas.

**Métricas** (script versionado `tools/metricas.py`, lendo o `_coords.json` e o JSON de legenda):
- câmera: movimentos por minuto, reversões por segundo, deslocamento em px/s, trocas de layout por minuto;
- limites: cortes que terminam no meio da frase, cortes que começam/terminam no meio de uma palavra;
- tempo por etapa (download, transcrição, IA, corte, enquadramento, legenda) e por corte (decodificação, detecção, composição, encode).

| Ordem | Commit | O que muda | Métrica de aceite |
|---|---|---|---|
| 1 | Medição | Tempo por etapa no log + `timings.json`; tempo por corte no enquadramento; `tools/metricas.py` | linha de base registrada |
| 2 | Velocidade rápida | InsightFace só com detecção (+ landmark 3D quando "focar em quem fala" está ligado); mux com `-c:v copy`; download preferindo H.264 | detecção −56%/−85%; 3 encodes por corte em vez de 4 |
| 3 | IA robusta | `google-genai` (timeout, raciocínio baixo, JSON com esquema), cota diária × por minuto, retentativa só em códigos transitórios, `json-repair`, heurística de reserva, contagem de requisições no log; SDK antigo como fallback | nenhum travamento silencioso; erro de cota com mensagem clara |
| 4 | Hook | Esquema e prompt com `hook` ≤ 10 palavras, diferente do título | hook ≠ título em 100% dos cortes |
| 5 | Câmera v2 | Item 1 da lista; parâmetros por tipo de vídeo | movimentos/min e reversões/s menores nos clipes do Flow e do GTA |
| 6 | Respiro e áudio | Início até 0,25 s antes da 1ª palavra (sem pegar a anterior), fim 0,35–0,5 s após a última (sem pegar a próxima), fade 80/250 ms, loudnorm −14/TP −2 | 0 cortes no meio de palavra |
| 7 | Legendas | Quebra por pontuação/pausa, limite de caracteres, sem palavra solta, bloco até a próxima fala | 0 blocos de 1 palavra (fora interjeições) |
| 8 | Paralelismo | N cortes ao mesmo tempo (padrão 2 no Colab), um processo por corte | tempo do enquadramento menor (a medir: o Colab grátis tem só 2 CPUs) |
| 9 | Opcionais por tipo | Punch-in (podcast, entrevista, aula); compressão de pausas segura (podcast, aula); remoção de vícios experimental (desligada) | punches ≈ 2–4/min; nenhuma pausa removida durante ação |
| 10 | Notebook 3.0 | `tools/build_notebook_v3.py` gera `ViralCutter_v3.ipynb` (4 células, clona o fork em `feat/edicao-humanizada`, instalação fixada + `google-genai` + `json-repair`, opções novas na célula 2, tempo de cada etapa no log, resumo no topo) | gerado pelo script versionado |

**Fora do plano, e por quê:**
- render em CUDA ou por `sendcmd`: alto esforço; decido depois de medir o paralelismo;
- escolha de layout ou hook por imagens no Gemini: gasta cota (20/dia no Flash);
- YOLO (AGPL), AssemblyAI (pago), pycaps (precisa de Chromium).

**Dependências novas:** `google-genai` e `json-repair`. As duas resolvem no venv local com as versões fixadas
(`protobuf<5`, `numpy<2`, `huggingface-hub<1.0`) sem trocar nenhum pacote. A confirmação final é na sua
rodada do Colab.
