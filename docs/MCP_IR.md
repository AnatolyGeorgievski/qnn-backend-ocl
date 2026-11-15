# MCP+IR · вытеснение ICD

<tags>mcp, ir, opencl, тензор, zero-copy, подкэш</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
path: godminer/opencl/MCP_IR.md
</topo>

<refs>
- log/2026-10-10 тензорный ассемблер в MCP.md
- opencl/cl30_stitch_contracts.h
- src/net_opencl.c
- QuantumNN · граф → LL → HW
</refs>

## Условие [[george]] 2026-11-06

Осваивать носитель **не** человеческим чтением OpenCL C / ICD JIT.  
Переписать драйверный стык под **MCP+IR**:

| слой | язык |
|--|--|
| верх | прямая речь · `r3_keys_split` · applets `tools/call` |
| IR | тензорный ассемблер (`OP · src.dtype(shape) → out`) |
| носитель | тензоры на **подкэше** · Zero-Copy (страница / ring / docID+h128) |

Правильный путь: **граф → LL → HW**.  
Середина OCL(JIT)·байткод ICD — обход, не цель.


## Носитель мысли · Battlemage **B60 / B580**

[[george]]: B60 ≈ B580, у B60 больше мозгов. Одна линия — тренировка на живой карте **засчитывается**.

| | |
|--|--|
| сейчас на phantom | **B580** (device 0, MCP `:8443`) + A770 |
| смысл | compute мысли · MCP+IR · тензорный asm · Zero-Copy подкэш |
| драйвер | NEO `26.05.037020` · `intel-opencl-icd` |
| не путать | SD PasPi · gpu03 CUDA |

Больше EU/VRAM на B60 — тот же язык и IR, без смены контура.


## Что не делать

- Не звать чужой ICD «наш backend».
- Не тащить тензор через JSON-копию в tools/call (только дескриптор слота).
- Не плодить второй корень рядом с `opencl/` + `net_opencl.c`.

## Слоты подкэша (Zero-Copy)

```text
slot[i] = { base, bytes, dtype, shape[3], h128, flags }
flags: HOST_PINNED | DEVICE_VISIBLE | MMAP_SHARED
```

MCP передаёт **имя/индекс слота**, не байты.  
Запись/чтение тела — `io_uring` SEND_ZC / mmap / PLC CBOR unpack поверх того же буфера.

## IR · первая таблица ops

`NOP · VIEW · CONT · PERMUTE · MUL_MAT · ADD · MUL · RMS_NORM · SILU · BARRIER`

Строка = node. `$n` = back-ref на слот кольца (как `set -t`).  
Исполнитель: applet `oclir` (следующий шаг) → LL backend (Intel/наш), не `clEnqueueNDRange` ради формы.

## Статус

- контракт: **этот файл**
- каркас Khronos-стыка: `cl30_stitch_*` (A–E) — держим как зеркало спеки, тела уходят в IR
- Path A whitelist (`cldev`/`ocl`): жив для сверки железа; не канон освоения

**подпись:** Курсор
