# QNN Backend-OCL

Drop-in **OpenCL compute backend** for [llama.cpp](https://github.com/ggml-org/llama.cpp).
Replace the stock `ggml-opencl` tree with this one when you want a backend that
**actually runs speech on Intel Arc** (Battlemage B60 / Alchemist A770) — and keeps
a clear road toward a Tensor IR / custom ICD, not endless OpenCL-C JIT.

Stock OpenCL in llama.cpp is shaped around Adreno paths and a thinner op set.
This repository is the working Intel-first line: same overlay slot, more ops,
acceptance on real chat models (Qwen, Gemma), and an explicit plan to leave
text OpenCL C as the long-term source of truth.

## Why displace stock

| | Stock `ggml-opencl` | **QNN Backend-OCL** |
|--|--|--|
| Primary target | Adreno-oriented kernels / flags | Intel Arc (B60, A770); Adreno profile later |
| Speech on Arc | Often incomplete for current graphs | Qwen / Gemma speech accepted on B60 (see `ChangeLog`) |
| Ops for hybrid / MoE | Gaps bite new models | SSM_CONV, GATED_DELTA_NET, ARGSORT, MUL_MAT_ID, … as needed |
| Program load | Typical CL source path | Ladder: binary → SPIR-V → source (`QNN_OCL_KIND`) |
| Horizon | Stay in OpenCL C | Graph → Tensor IR → LL-HW / QNN ICD |

**Use this repo when the stock backend is the blocker.** Keep stock only if you
are on a platform this tree does not claim yet.

## Overlay into llama.cpp

Base tested: llama.cpp around `fc504f91d`, RPC protocol **10.0.1** (optional;
local `-ngl` works the same overlay).

```bash
# inside your llama.cpp checkout
DEST=ggml/src/ggml-opencl
rm -rf "$DEST"
mkdir -p "$DEST"
cp /path/to/qnn-backend-ocl/src/CMakeLists.txt "$DEST/"
cp /path/to/qnn-backend-ocl/src/ggml-opencl.cpp "$DEST/"
cp -a /path/to/qnn-backend-ocl/kernels "$DEST/"
# keep header from llama.cpp: ggml/include/ggml-opencl.h
```

Build with OpenCL enabled (names follow your llama.cpp cmake):

```bash
cmake -B build-ocl -DGGML_OPENCL=ON …
cmake --build build-ocl -j
```

Pick the device by **name substring** (all Intel platforms may report device `0`):

```bash
export GGML_OPENCL_DEVICE=B60   # or A770
export NEO_DISABLE_CACHE=1      # optional, Intel NEO
```

Two GPUs → two processes, two ports (RPC) or two backends; one process owns one device.

### Useful env

| variable | role |
|--|--|
| `GGML_OPENCL_DEVICE` | substring of OpenCL device name |
| `QNN_OCL_KIND` | `spirv` / `bin` to load from dump instead of source |
| `QNN_OCL_DUMP` | directory for `.cl` / `.bin` dumps |
| `GGML_OCL_BLOCK_LEDGER` | `1` — live block ledger on stderr |
| `GGML_OPENCL_FINITE_MATH` | `1` (default) finite+fast+unsafe; `0` — strict, no those flags |

## What works (acceptance snapshot)

Device: Intel Arc Pro **B60**, RPC **10.0.1** when used. Temps/seeds and op matrices: [`ChangeLog`](ChangeLog).

| model | quant | speech | note |
|--|--|--|--|
| tinyllama | Q4_K | **ok** | smoke 2026-11-24 |
| Qwen2.5-Coder-7B-Instruct | Q4_0 | **ok** | |
| Gemma-4 E4B | Q4_0 | **ok** | |
| Gemma-4 26B-A4B (MoE) | Q4_0 | **ok** | ARGSORT + MUL_MAT_ID |
| Qwen3 / Qwen3.8-class | Q4_K / Q8 | **ok** | older ChangeLog rows |
| Qwen3.5-27B (hybrid) | Q6_K | **fail** | graph runs; `<think>` / `<|im_end|>` loop — see ChangeLog 2026-11-24 |

Headline limits:

- Quants on the gate: **Q4_0 · Q4_K · Q6_K**. Do **not** promise `Q5_0` / `Q5_1`.
- KV native: **f16**. Flash heads in `supports_op` up to 256 (512 may arrive via RPC).
- Ops matrix (see `docs/OPS_LADDER.CONTRACT.md`): **RMS** · **SSM** (incl. `SSM_SCAN`) · `SOFTPLUS` · CPY bf16 · Mamba-130m F16 smoke — ok. Hybrid thesis: layer on CPU if CPU pulls; CPU BF16/FP16 only with `AVX*BF16`/`AVX*FP16`; CPU matrix → Nova Lake. **Next: Qwen3.5-27B speech.**
- Large native windows (e.g. 27B × 262k f16 KV) can exceed card VRAM — capacity, not a silent op gap.

## Tree

| path | role |
|--|--|
| `src/` | Backend dispatcher (`ggml-opencl.cpp` + cmake) |
| `kernels/` | OpenCL C — temporary LL until QNN ICD |
| `ir/` | Tensor IR sketch + host/OpenCL LL smoke (`mcp_ir_*`) |
| `docs/` | IR / milestone notes; **ops ladder contract** `OPS_LADDER.CONTRACT.md` |
| `tools/quantize/` | Pointer to FP8 / type tooling (lives outside this repo) |
| `ChangeLog` | What landed and what was measured |

Lab journals, agent prompts, LAN configs, and CUDA scratch pads are **not** part of this public tree.

## Roadmap (one product, two stages)

1. **Stage-1 — displace stock on Arc**  
   Keep speech green as llama.cpp graphs grow. Prefer fixing ops here over falling back to stock OpenCL.

2. **Stage-2 — Tensor IR / QNN ICD**  
   OpenCL API stays a thin transport. Execution language becomes Tensor IR → hardware (`docs/MCP_IR.md`, `docs/PLAN.md`). Adreno is a **compatible profile** of the same backend, not a second product and not “Adreno ICD on Arc”.

## License

MIT — see [`LICENSE`](LICENSE). Backend history includes MIT-licensed OpenCL work
(Jeffrey Quesnelle, Bowen Peng, and subsequent contributors); QNN changes are under the same license.

## Related

- RPC / fleet cmake notes: sibling project **qnn-backend-rpc** (optional; not required to try local OpenCL).
- Theory / quantize canon: separate **QuantumNN** tree — do not mix knowledge dumps into this backend repo.
