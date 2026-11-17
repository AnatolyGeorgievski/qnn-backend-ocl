# QNN Backend-OCL

<tags>
- src/ggml-opencl.cpp
- kernels/
- ChangeLog
- ir/mcp_ir.h
- tools/quantize/
- docs/MCP_IR.md
- docs/PLAN.md
- log/
- log/2026-11-15 qwen конфиг arc3.md
- log/qwen/
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
| quant | `tools/quantize` — BF16/F16 → свой f8 (в составе этого проекта) |

### 2. Поддержка Мити · Adreno

После цели 1 — **тот же** backend или **совместимый** профиль под Adreno для [[lumag]] (Митя).

- Не «Adreno ICD на Arc» и не эмуляция Adreno поверх Intel.
- Прогон на Adreno — только при ходе Мити, **его** железе (туннель) и его постановке драйвера; цель — отладить наш backend / доработать стык под его драйвер.
- Пресеты `opencl-arc` / `opencl-adreno` не смешивать (см. QuantumNN/SAT.md · урок 2).
- Совместимость: общий Tensor IR + те же ops; LL-HW / ICD-профиль — отдельная половина двери `ADRENO`, не развилка продукта.

## Сборка на llama.cpp 10.0.1

Backend кладётся в дерево llama.cpp поверх патча RPC. База — `fc504f91d`. Патч — `qnn-backend-rpc/notes/patches/0002-llama-cpp-10.0.1.patch`, протокол `10.0.1`. Как наложить патч — в README того репозитория. Заголовок `ggml/include/ggml-opencl.h` остаётся из llama.cpp.

Каталог `ggml/src/ggml-opencl` после патча заменяется файлами отсюда: `src/CMakeLists.txt`, `src/ggml-opencl.cpp`, каталог `kernels/`.

Пресет `opencl-arc` пишет в `build-opencl-arc`. Эта сборка идёт в отдельный каталог `build-our-ocl`, теми же флагами:

```bash
OCL=/path/to/qnn-backend-ocl
DST=ggml/src/ggml-opencl
rm -rf "$DST"
mkdir -p "$DST"
cp "$OCL/src/CMakeLists.txt" "$OCL/src/ggml-opencl.cpp" "$DST/"
cp -a "$OCL/kernels" "$DST/kernels"

cmake -S . -B build-our-ocl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_RPC=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_OPENCL_SOA_Q=ON \
  -DGGML_OPENCL_PROFILING=OFF \
  -DGGML_OPENCL_TARGET_VERSION=300 \
  -DLLAMA_CURL=OFF
cmake --build build-our-ocl -j --target llama-cli ggml-rpc-server
```

Нужны заголовки OpenCL, Ninja и Python 3: ядра вшиваются `kernels/embed_kernel.py`. Ядра Adreno выключены.

Сервер из `build-our-ocl/bin`:

```bash
NEO_DISABLE_CACHE=1 LD_LIBRARY_PATH=$PWD ./ggml-rpc-server -H 0.0.0.0 -p 18769
```

Клиент из того же дерева. Версия клиента и сервера — `10.0.1`.

```bash
LD_LIBRARY_PATH=build-our-ocl/bin ./build-our-ocl/bin/llama-cli \
  --rpc HOST:18769 -dev RPC0 -ngl 99 \
  -m model.gguf -p 'Hello' -n 16 --temp 0 -s 1 -c 512 -st
```

Баннер: `Starting RPC server v10.0.1`. Устройство на прогоне: Intel Arc Pro B60, OpenCL 3.0.

## Что держит эта сборка

Очередь команд идёт в порядке подачи. KV для этого backend — F16, активация остаётся float. На устройстве перекладывается только `Q4_0`. `Q8_0`, `F16` и остальные типы пишутся теми байтами, что пришли с хоста.

17 ноября 2026, Qwen3-8B Q8_0, температура 0, семя 1, текст про прямую речь и benchmark: первая фраза совпала с CUDA 10.0.1. Промпт 66.7 t/s, генерация 33.4 t/s.

Приёмка этой сборки — Qwen и Gemma. Заявка, которую можно показывать: речь Qwen на Arc Pro B60 через этот backend и RPC 10.0.1.

Tensor IR и свой ICD остаются целью репозитория. Граф этой сборки исполняет OpenCL C через Intel ICD. Слоты `get_tensor_async` и `cpy_tensor_async` пустые. Пачка `set_tensor_n` в прогоне речи не вызывалась: вес приходит по одному тензору. `BF16` KV эта сборка не обслуживает: ядра `set_rows` для него нет.

## Дерево

| путь | что |
|--|--|
| `src/` | Stage-1 диспетчер графа (архив inbox) |
| `kernels/` | OpenCL C — временный LL до ICD QNN |
| `ir/` | Tensor IR (рабочая копия) |
| `tools/quantize/` | конвертер весов + отчёт ошибки по тензору |
| `docs/` | MCP_IR, план вех |
| `log/` | журнал; конфиг Qwen/arc3 |


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
