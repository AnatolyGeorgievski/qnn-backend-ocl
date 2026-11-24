# BLACKWELL-SELF · qwen на RTX PRO 4000 (sm_120a)

## 1. Метод (свой)

Шёл от скрипта к backend, не от документации:

1. `qwen-server-start.sh` → `qwen-server.env` — что реально запускается и с какими env.
2. `main.cpp` → `llama_server` (server.cpp ~798) → `server_run_instance` → `load_model` (server-context.cpp ~1204) → `common_init_from_params` (common.cpp ~1432) — где backend выбирается и куда уходит граф.
3. Сборка: `log/llama-cpp-build.env` — зафиксировал, что компилируется **только** `CMAKE_CUDA_ARCHITECTURES=120a` (nvcc 13.2, driver 595.99.02, GPU RTX PRO 4000 Blackwell, COMPUTE_CAP=12.0). Значит весь путь — self-run: один arch, один бинарь, без fallback-сборок.
4. В `ggml-cuda.cu` читал только 4 окна (init / graph_compute / supports_op / features) + `blackwell_mma_available` в `common.cuh` — чтобы увидеть, где именно 120a меняет поведение.

## 2. Каркас вызовов (строки из ggml-cuda.cu)

- `ggml_cuda_init()` — **258–360**: `cudaGetDeviceCount` (261), `cudaGetDeviceProperties` (301, 334), `nsm/smpb/warp_size` (346–348), VMM (318–331). Для 120a: `cc` = 1200, `GGML_CUDA_CC_BLACKWELL`.
- `ggml_backend_cuda_graph_compute()` — **5422–5485**: `USE_CUDA_GRAPH` (5431), warmup-логика (5444–5467), `cudaStreamBeginCapture` (5479), `ggml_cuda_graph_evaluate_and_capture` (5482). В сборке `GGML_CUDA_GRAPHS=ON` — путь активен.
- `ggml_backend_cuda_device_supports_op()` — **6199–6300**: `cc_dev` (6275), `f32_src1_ok` (6276), `fp8_id_mmvf` (6278–6280). Для 120a: `cc_dev >= GGML_CUDA_CC_AMPERE` → f32 src1 ок; FP8 MMA доступен (cc ≥ ADA_LOVELACE).
- features — **6820–6850**: `has_fp8 |= fp8_mma_hardware_available(cc)` (6830), `has_fp4 |= blackwell_mma_available(cc)` (6831) → `BLACKWELL_NATIVE_FP4=1` (6840).
- `blackwell_mma_available(cc)` — `common.cuh` **394–397**: `GGML_CUDA_CC_IS_NVIDIA(cc) && highest_compiled_arch(cc) >= 1200 && < RUBIN`.

## 3. Только Blackwell / 120a

- `GGML_CUDA_CC_BLACKWELL = 1200` (common.cuh:60).
- `blackwell_mma_available` (common.cuh:394) — единственный гейт на native FP4 (NVFP4/MXFP4): `mmq.cu:127` (`can_use_q4`), `mmq.cuh:255`.
- `BLACKWELL_NATIVE_FP4` feature (ggml-cuda.cu:6840) — включается только при 120a.
- `fattn.cu:407` — `cc >= GGML_CUDA_CC_BLACKWELL` (отдельный FA-путь).
- `mmvf.cu:948` — `blackwell = cc >= GGML_CUDA_CC_BLACKWELL`.
- `mmvq.cu:331` — `cc == GGML_CUDA_CC_BLACKWELL` (ровно 1200, не выше).
- `gated-delta-net-mma.cu:405` — `cc != GGML_CUDA_CC_BLACKWELL && cc != GGML_CUDA_CC_DGX_SPARK` → reject.
- common.cuh:314 — `__CUDA_ARCH__ >= 1200 && < RUBIN` (device-side guard).
- Сборка: `CMAKE_CUDA_ARCHITECTURES=120a` — self-run, один arch; `GGML_CUDA_FA_QUANTS` включает `f8_e4m3-f8_e4m3` (FP8 FA на 120a).

## 4. Вне контура

- OpenCL `ggml-opencl.cpp` — `GGML_OPENCL=OFF`, не компилируется, не участвует.

DONE
