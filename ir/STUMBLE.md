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
MUL_MAT ocl LL закрыт, в т.ч. на B60 (phantom). Следующий пазл — отдельно.

## Приёмка RPC+OCL · 2026-11-21

phantom `192.168.1.185:18769`, `GGML_OPENCL_DEVICE=B60`, сервер `llama-our-ocl-1001` v10.0.1.
Клиент: arc3 `build-cuda-rpc` llama-cli → `--rpc 192.168.1.185:18769 -dev RPC0`.

| модель | результат |
|--|--|
| tinyllama Q4_0 | OK, речь |
| gemma-4-E4B Q4_0 | OK, «Hello! How can I help you today?» · 38.8 / 20.3 t/s |
| Qwen2.5-0.5B | FAIL: `Qcur-0` MUL_MAT not supported (не приёмка) |

arc2 без GPU — OpenCL там не ждать. Pid: `/home/george/rpc-server.pid` на phantom.

## Приёмка RPC+CUDA · 2026-11-21

arc3 `192.168.1.187:18767`, `ggml-rpc-server -d CUDA0,CUDA1`, v10.0.1.
Клиент: тот же `build-cuda-rpc/bin/llama-cli`.

| модель | результат |
|--|--|
| tinyllama Q4_0 | OK · ~468 / 250 t/s |
| gemma-4-E4B Q4_0 | OK · «Hello! How can I help you today?» · ~90 / 61 t/s |

Пара OCL `:18769` (B60) и CUDA `:18767` — один протокол 10.0.1.
