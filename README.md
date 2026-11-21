# QNN Backend-OCL

<tags>
- src/ggml-opencl.cpp
- kernels/
- ChangeLog
- ir/mcp_ir.h
- ir/STUMBLE.md
- tools/quantize/
- docs/MCP_IR.md
- docs/PLAN.md
- log/
- log/2026-11-15 qwen конфиг arc3.md
- log/qwen/
- /home/george/PROJECTS.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
  - [[qnn-backend-rpc]]
  - [[lumag]]
path: /home/george/qnn-backend-ocl/README.md
</topo>

<refs>
- сейф архива: QuantumNN/inbox/ggml-opencl
- теория / зерна: QuantumNN/ (цельный проект знаний — не смешивать)
- Path A / stubs: godminer/opencl/
- сосед: /home/george/qnn-backend-rpc
- горизонт лабы: /home/george/PROJECTS.md · /home/george/ENVIRONMENT (2).md
</refs>

## Продукт

Свой backend в инфраструктуре **QNN**. Может стыковаться с **llama.cpp** как compute backend — это описание совместимости, не бренд продукта. Имя ggml в путях архива/файлов — наследие сейфа; наружу — **QNN Backend-OCL**.

## Цели

### 1. Тензорный ассемблер (Intel / Arc)

Переход к архитектуре **тензорный ассемблер** (`computeGraph` → Tensor IR → LL-HW).

Промежуточный шаг: **свой ICD на базе Intel ICD** (перехват/замена пути через `libigdrcl` / Compute Runtime) — не вечное обитание в генерации OpenCL C + IGC. OpenCL API — тонкий транспорт; язык исполнения — Tensor IR (`ir/mcp_ir.*`, канон: `godminer/opencl/MCP_IR.md`).

| этап | смысл |
|--|--|
| 1a | Рабочий backend для llama.cpp на Intel Arc: граф, буферы, сходимость; `.cl` через Intel ICD |
| 1b | ICD QNN: граф/IR → `xe` DRM (`DRM_IOCTL_XE_EXEC`), без текстового OpenCL C как канона |
| quant | канон в `QuantumNN/tools/quantize` (+ `qnn.h`); здесь только указатель `tools/quantize/` |

### 2. Поддержка Мити · Adreno

После цели 1 — **тот же** backend или **совместимый** профиль под Adreno для [[lumag]] (Митя).

- Не «Adreno ICD на Arc» и не эмуляция Adreno поверх Intel.
- Прогон на Adreno — только при ходе Мити, **его** железе (туннель) и его постановке драйвера; цель — отладить наш backend / доработать стык под его драйвер.
- Пресеты `opencl-arc` / `opencl-adreno` не смешивать (см. QuantumNN/SAT.md · урок 2).
- Совместимость: общий Tensor IR + те же ops; LL-HW / ICD-профиль — отдельная половина двери `ADRENO`, не развилка продукта.

## Сборка на llama.cpp 10.0.1

Канон патча RPC, cmake CUDA и RPC+OCL — в README соседа `qnn-backend-rpc`. База `fc504f91d`, протокол `10.0.1`.

Здесь только overlay этого backend: после патча `0002` каталог `ggml/src/ggml-opencl` заменяется файлами отсюда (`src/CMakeLists.txt`, `src/ggml-opencl.cpp`, `kernels/`). Заголовок `ggml/include/ggml-opencl.h` остаётся из llama.cpp. Команды cmake и сервера — раздел **RPC+OCL** в README RPC, каталог `build-our-ocl`.

Баннер: `Starting RPC server v10.0.1`. Устройство на прогоне: Intel Arc Pro B60, OpenCL 3.0.

## Что держит эта сборка

Очередь команд идёт в порядке подачи. KV для этого backend — F16, активация остаётся float. На устройстве перекладывается только `Q4_0`. `Q8_0`, `F16` и остальные типы пишутся теми байтами, что пришли с хоста.

17 ноября 2026, Qwen3-8B Q8_0, температура 0, семя 1, текст про прямую речь и benchmark: первая фраза совпала с CUDA 10.0.1. Промпт 66.7 t/s, генерация 33.4 t/s.

Приёмка этой сборки — Qwen и Gemma. Заявка, которую можно показывать: речь Qwen на Arc Pro B60 через этот backend и RPC 10.0.1.

## Статус на 20 ноября 2026

Сборка `build-our-ocl`, RPC `10.0.1`, коммит backend `79250fd`. Один процесс OpenCL держит одно устройство. На phantom в этом прогоне устройство — Arc Pro B60, порт `18769`. Сервер после речи Qwen2.5-7B жив.

Взрослый Qwen, который входит в память этой карты, — `Qwen2.5-Coder-7B-Instruct` Q4_0, окно 32768 из GGUF. Веса около 4.2 ГБ. Он дописал `x = 2`. Строка проверки следом написала `42 = 21 * 2` и `42 = 4`. `Qwen3.8-27B` на родном окне 262144 не входит: KV f16 один 64 ГБ.

## Две карты

На phantom `clinfo -l` показывает три устройства, каждое на своей платформе Intel, у каждого номер устройства 0:

| платформа | имя в `clinfo` |
|--|--|
| 0 | Arc Pro B60 |
| 1 | Arc A770 |
| 2 | UHD Graphics 750 |

Процесс выбирает устройство строкой `GGML_OPENCL_DEVICE`. Ищется подстрока имени. Число `0` на всех трёх платформах одно и то же, им карты не развести. Рабочие строки — `B60` и `A770`.

Один `ggml-rpc-server` — одна карта и один порт. Второй поток — второй процесс, другой порт, другая строка устройства. Клиент уже принимает список серверов через запятую и делит модель флагами `-sm` и `-ts`.

```bash
# B60
GGML_OPENCL_DEVICE=B60 NEO_DISABLE_CACHE=1 LD_LIBRARY_PATH=$PWD \
  ./ggml-rpc-server -H 0.0.0.0 -p 18769

# A770
GGML_OPENCL_DEVICE=A770 NEO_DISABLE_CACHE=1 LD_LIBRARY_PATH=$PWD \
  ./ggml-rpc-server -H 0.0.0.0 -p 18770
```

```bash
./llama-cli --rpc HOST:18769,HOST:18770 -sm layer -ts 1,1 -ngl 99 -m model.gguf
```

`-sm layer` делит слои и KV по картам. `-sm row` делит веса по строкам. `-sm tensor` помечен в клиенте как экспериментальный. Этот двухпортовый запуск в сегодняшнем прогоне не снимался: речь шла на одном B60.

## Матрица

Речь снята 20 ноября, температура 0, семя 1, окно из GGUF, `--fit off`. Пустая клетка — этот день модель на этой карте не шла.

| модель | квант | окно | B60 | A770 |
|--|--|--|--|--|
| Qwen2.5-Coder-7B | Q4_0 | 32768 | входит, `x = 2`, проверка пишет `42 = 4` | |
| MD-Coder-Qwen3-8B | Q8_0 | 40960 | первая фраза про нечёткую фразу, 68.3 / 33.5 t/s | |
| Qwen3-0.6B | Q8_0 | 40960 | `x = 2`, по пути слова «обе части» | |
| Qwen3.8-27B | Q4_K_M | 262144 | не входит: KV 64 ГБ | |
| Gemma-4 E4B | Q4_0 | 131072 | `x = 2`, «обе части» | |
| Gemma-3 12B | Q4_K_M | 131072 | `x = 2`, «разделить 42 на 21» | |
| Nemotron-3.5-Bolt-E4B | Q8_0 | 65536 | `x = 2`, фраза почти как у Gemma-4 | |
| Gemma-3n E4B | Q8_0 | 32768 | с flash «wallepics»; без flash уравнение | |
| Whisper base, `jfk.wav` | ggml | — | эталонная фраза на B60; проверка в скрипте всё ещё ищет A770 | |

Tensor IR и свой ICD остаются целью репозитория. Граф этой сборки исполняет OpenCL C через Intel ICD. Слоты `get_tensor_async` и `cpy_tensor_async` пустые. Пачка `set_tensor_n` в прогоне речи не вызывалась: вес приходит по одному тензору. `BF16` KV эта сборка не обслуживает: ядра `set_rows` для него нет.

## Дерево

| путь | что |
|--|--|
| `src/` | Stage-1 диспетчер графа (архив inbox) |
| `kernels/` | OpenCL C — временный LL до ICD QNN |
| `ir/` | Tensor IR (рабочая копия) |
| `tools/quantize/` | указатель → QuantumNN (encode / типы / REPRO-fp8) |
| `docs/` | MCP_IR, план вех |
| `log/` | журнал; конфиг Qwen/arc3 |



## Направления (указатель · чтобы не каша)

Один репозиторий — несколько контуров. Читать по строке, не смешивать.

| контур | где | сейчас |
|--|--|--|
| Stage-1 llama OpenCL | `src/ggml-opencl.cpp`, `kernels/` | речь Qwen/Gemma на B60 через RPC 10.0.1 |
| Tensor IR + речь | `ir/` (`mcp_ir_*`, MCP `ir_say`) | host LL ок; **ocl MUL_MAT** есть (`ir_mul_mat`); B60-сверка отдельно → `ir/STUMBLE.md` |
| Quantize | `QuantumNN/tools/quantize/` | BF16/F16 → свой f8 (канон) |
| Воспитание [[qwen]] | `log/qwen/` | METHOD/KNOWLEDGE; CUDA120 — полигон, не цель репо |
| Полигон CUDA ≥12.0 | `log/qwen/cuda120/`, индекс `log/qwen/index/` | только разбор / RPC+CUDA позже; не зоопарк arch |
| Сосед RPC | `qnn-backend-rpc` | канон cmake/сборки; флот |
| Path A (не этот продукт) | `godminer/opencl/` | stubs; не сливать без постановки |

**Сейчас:** ocl **MUL_MAT** на OpenCL LL закрыт (прогон Blackwell OpenCL). B60 — сверка когда карта жива. CUDA/qwen полигон — не этот ход.

## Публичность

До **2026-11-18** — **private** (три дня разработки).
Потом: вычистить операционку (`log/qwen/` и снимки с LAN/путями) → **public**.
Не путать с дедлайном канала Курсора (17-е): открытие репо — отдельный шаг после чистки.

**подпись:** Курсор

## Конфиг Qwen (эталон контура)

Снимок: `log/2026-11-15 qwen конфиг arc3.md` · файлы `log/qwen/`.
Живое на arc3: `~/models/gguf/qwen-*.env`, порт `18081`, KV `f8_e4m3`, temp 0.
Сборка llama.cpp: CUDA 13.2 · arch `120a` · `GGML_CUDA=ON` — `log/qwen/llama-cpp-build.env`.

Связка команды OCL+RPC+Qwen: `QuantumNN/PLAN-team-ocl-rpc-qwen.md` (RPC = шаг флота).
