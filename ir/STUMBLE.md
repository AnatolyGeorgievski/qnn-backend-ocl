# IR · hard edges

## Host LL

ADD / MUL / SUB / DIV / NEG / SILU / RMS_NORM / **MUL_MAT** — speech converges (`make` in `ir/`).

## OpenCL LL

Elementwise ops and RMS match host.
**MUL_MAT** on OpenCL LL: `ir_mul_mat` + `mcp_ir_ocl_mul_mat`.
Checked on Intel Arc OpenCL and on discrete NVIDIA OpenCL where available.

## Backend acceptance (RPC+OCL, 2026-11-21+)

Device: Intel Arc Pro **B60**. Protocol 10.0.1 when using RPC.

| model | result |
|--|--|
| tinyllama Q4_0 / Q4_K / Q6_K | OK |
| gemma-4 E4B Q4_0 | OK speech |
| Qwen2.5-Coder-7B Q4_0 | OK speech |
| Qwen3.5-27B Q6_K | hybrid path: CPY / SSM_CONV / SOFTPLUS / GDN unit; chat quality still a separate puzzle |
| Gemma MoE 26B-A4B Q4_0 | OK after ARGSORT + MUL_MAT_ID + CLAMP |

Quant types on the gate: **Q4_0 · Q4_K · Q6_K**. Do not advertise `Q5_0` / `Q5_1`.

CUDA RPC on the same protocol is a dual-path check (same client, same phrase);
OCL is slower but text can match.

## Notable fixes kept in tree

- `CPY` into views: accept `cpy_form` / view ancestry (was false reject).
- `SSM_CONV`, `SOFTPLUS`, `GATED_DELTA_NET`, `L2_NORM` for hybrid Qwen graphs.
- `ARGSORT` + `MUL_MAT_ID` for Gemma MoE.
- `get_alloc_size`: avoid over-padding that trips harness asserts.
- `rms_norm_mul` fusion: epsilon from the RMS_NORM node.

Full op matrices and model tables: root `ChangeLog`.

## Hybrid Qwen3.5 speech · repro 2026-11-24

Server: phantom `llama-our-ocl-1001` RPC `:18769`, `GGML_OPENCL_DEVICE=B60`.
Client: arc3 `llama-cli` `0cce758`, model `Qwen3.5-27B-Q6_K`.

| flags | visible output | rate |
|--|--|--|
| default / jinja (chat) | empty (specials hidden) | gen ~0 or silent EOS |
| `--jinja --reasoning off -sp` | first token **`<\|im_end\|>`** | ~7.8–7.9 t/s |
| `--no-jinja --reasoning off` | loop **`<think>`** | ~8.2 t/s |

Graph runs (prompt ~4–5 t/s). Not a missing-op crash. Same class of bug as 2026-11-22 smoke.
CUDA dual-path today: RPC `:18767` not ready (GPU busy with local llama-server).

Next cut: first-token logits OCL vs CUDA/CPU on the same prompt (free CUDA slot or smaller hybrid).

