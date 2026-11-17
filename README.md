# QNN Backend-OCL

<tags>
- src/ggml-opencl.cpp
- kernels/
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
