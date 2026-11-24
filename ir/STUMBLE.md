# IR · где споткнулись

<tags>
- ir/mcp_ir.c
- ir/mcp_ir_ocl.c
- ir/mcp_ir_mcp.py
</tags>

## Host LL
ADD/MUL/SUB/DIV/NEG/SILU/RMS_NORM/**MUL_MAT** — речь сходится (`make ir`).

## OpenCL LL
Элементные ops и RMS — совпали с host.
**MUL_MAT на ocl LL** — `ir_mul_mat` + `mcp_ir_ocl_mul_mat`; речь через `ll_mul_mat`.
Прогон 2026-11-21: Blackwell OpenCL (arc3) → OK; **B60 на phantom `192.168.1.185`** → OK (`1 0 1 0`).
arc2 без GPU в PCI — 0 platforms нормален для этой машины.

## Qwen
Чат: `http://192.168.1.187:18081/` · MCP `fetch` + `mcp-ir` (`ir_say`).

## Сейчас
Очередь: `log/qwen/ROAD.md` — **координация**; OCL-речь 27B **отложена**.
Gemma MoE 26B-A4B на ocl: ARGSORT · MUL_MAT_ID · CLAMP — речь ок.
27B: CPY · SSM_CONV · SOFTPLUS · GDN unit ок; речь (`<|im_end|>`) — не чинить в этом такте.
Таблица приёмки 2026-11-21 ниже — история; актуальный стоп см. этот блок и ROAD.

## Приёмка RPC+OCL · 2026-11-21

phantom `192.168.1.185:18769`, `GGML_OPENCL_DEVICE=B60`, сервер `llama-our-ocl-1001` v10.0.1.
Клиент: arc3 `build-cuda-rpc` llama-cli → `--rpc 192.168.1.185:18769 -dev RPC0`.

| модель | результат |
|--|--|
| tinyllama Q4_0 | OK, речь |
| gemma-4-E4B Q4_0 | OK, «Hello! How can I help you today?» · 38.8 / 20.3 t/s |
| tinyllama **Q4_K** pure | OK · 271.3 / 91.1 t/s |
| tinyllama **Q6_K** pure | OK · 237.8 / 91.1 t/s |
| Qwen2.5-Coder-7B Q4_0 | OK · «Hello! How can I assist you today?» · 90.8 / 27.6 t/s |
| Qwen3.5-27B Q6_K | CPY view OK после фикса; FAIL: `SSM_CONV` `conv_output_raw-0` |
| Qwen2.5-0.5B Q4_K_M | вне приёмки: внутри есть `q5_0` |

arc2 без GPU — OpenCL там не ждать. Pid: `/home/george/rpc-server.pid` на phantom.

## Приёмка RPC+CUDA · 2026-11-21

arc3 `192.168.1.187:18767`, `ggml-rpc-server -d CUDA0,CUDA1`, v10.0.1.
Клиент: тот же `build-cuda-rpc/bin/llama-cli`.

| модель | результат |
|--|--|
| tinyllama Q4_0 | OK · ~468 / 250 t/s |
| gemma-4-E4B Q4_0 | OK · «Hello! How can I help you today?» · ~90 / 61 t/s |

Пара OCL `:18769` (B60) и CUDA `:18767` — один протокол 10.0.1.

Кванты приёмки ocl: **Q4_0 · Q4_K · Q6_K** (и соседние K). `Q5_0` / `Q5_1` — не берём.

## Dual-path · 2026-11-21 19:18 UTC

Один клиент arc3 `llama-cli` 10.0.1, одна фраза `Hello`, `-n 48 --temp 0 -st -ngl 99`,
модель `/tmp/gemma-4-E4B-it-Q4_0.gguf`.

| путь | RPC | речь | prompt / gen |
|--|--|--|--|
| CUDA | `192.168.1.187:18767` | Hello! How can I help you today? | 180.6 / 65.7 t/s · wall 5.3s |
| OCL B60 | `192.168.1.185:18769` | Hello! How can I help you today? | 39.0 / 20.4 t/s · wall 29.4s |

Текст ответа совпал. На arc3 параллельно живёт llama-server (~20 GiB/GPU) — CUDA RPC всё равно вместился.

## Q5_0 · заметка · 2026-11-21

Qwen2.5-0.5B Q4_K_M: `Qcur-0` = MUL_MAT `src0=q5_0` × f32. `supports_op` раньше обещал Q5_0/Q5_1 без ветки — падение.
Ядро `kernel_mul_mat_q5_0_f32` на B60 речь давало (331.8 / 62.5 t/s) — откатили: тип редкий, тянуть через приёмку/валидации нет смысла.
Оставлено в логе на случай; в коде и `supports_op` — не держим.

## Масштаб B60 · 2026-11-21

| модель | результат |
|--|--|
| Qwen2.5-Coder-7B Q4_0 | OK · 90.8 / 27.6 t/s · wall 38.5s |
| Qwen3.5-27B Q6_K | CPY пройден; FAIL · `SSM_CONV` `conv_output_raw-0` · wall ~224s |

## CPY → view · 2026-11-21

`ggml_cpy` в `cache_r_l0 (view)`: `dst->view_src` схлопнут к корню, `src1` — промежуточный view → ложный отказ в `ggml_cl_cpy_check`.
Фикс: принять `cpy_form` (`dst->op == CPY`) и родство view_src. Gemma регрессии нет.
27B после фикса: CPY ок, стоп на **SSM_CONV** (Qwen3.5 recurrent) — отдельный пазл.

## SSM_CONV · 2026-11-21

Qwen3.5 / 3.8 hybrid: `linear_conv_kernel_dim=4` → `ggml_ssm_conv`. CUDA умел; ocl — нет.
Ядро `kernel_ssm_conv` в `ggml-opencl.cl`: окно по каналу, f32, смещения как у `SUM_ROWS`.
`supports_op`: сплошная строка канала, `ne[0]=d_conv-1+n_t`. Дальше на графе может вылезти `SSM_SCAN`.

## SSM_CONV · прогон · 2026-11-21

phantom B60 `:18769`, Qwen3.5-27B Q6_K, клиент arc3.
После CPY: **SSM_CONV** ок → **SOFTPLUS** (`a_softplus`) ок → FAIL: `GATED_DELTA_NET` `node_49`.

## Gemma 4 · 2026-11-22

| модель | ocl B60 |
|--|--|
| E4B Q4_0 | OK · та же фраза что CUDA |
| 12B Q4_0 (`ggml-org`) | OK · Hello! How can I help you today? · 18 / 13.8 t/s |
| 26B-A4B Q4_0 | OK · Hello! How can I help you today? · 28.1 / 20.9 t/s |

## ARGSORT · 2026-11-22

Bitonic `kernel_argsort_f32_i32` (как upstream ocl / CUDA). f32→i32, сплошной src, pad 2^n ≤ max_wg (B60: 1024).
Прогон 26B-A4B Q4_0 на phantom `:18769`: ARGSORT прошёл; стоп `MUL_MAT_ID` `ffn_moe_gate_up-0`.

## MUL_MAT_ID · CLAMP · 2026-11-22

`kernel_mul_mv_id_q4_0_f32_8x_flat` (SOA Q4_0, expert ids). После ARGSORT+MMID стоп был `CLAMP` workgroup (−54): local должен делить nelements.
Прогон 26B-A4B Q4_0 B60 `:18769`: речь «Hello! How can I help you today?» · 28.1 / 20.9 t/s.

## GATED_DELTA_NET · 2026-11-22

Ядро `kernel_gated_delta_net` (столбец=нить, S_v≤128) + host. Эталон CPU/CUDA.
`test-backend-ops` на B60: **43/43** GDN (без L2 в первом прогоне) OK — в т.ч. 128, GQA v_repeat, K>1, permuted, KDA.
SSM_CONV / SOFTPLUS / CONCAT / RMS_NORM / SCALE / FLASH hsk=256 / ROPE mode=40 — OK.
Smoke Qwen3.5-27B Q6_K B60 `:18769`: граф живёт, gen ~8 t/s, первый токен `<|im_end|>` (chat) или петля `<think>` (no-jinja). Gemma E4B на том же сервере — речь ок.
`get_alloc_size`: убран `GGML_PAD` (pad > nbytes ломал assert harness).
`rms_norm_mul` fusion: eps брать с узла RMS_NORM, не MUL.
`L2_NORM` ядро добавлено (разблокирует GDN+L2 тесты).
Дальше: состав hybrid-пути вне unit (state CPY / softplus×a / rms+scale на wide view) — пока GDN сам не виноват по NMSE.
