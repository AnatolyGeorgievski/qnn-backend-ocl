# Очередь · не терять

<tags>
- log/qwen/METHOD.md
- log/qwen/CHAT.md
- log/qwen/HELPER.md
- ir/STUMBLE.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
  - [[qwen]]
path: /home/george/qnn-backend-ocl/log/qwen/ROAD.md
</topo>

Порядок (2026-11-23):

1. ~~Агент-помощник~~ — контракт + ui-config (smoke моделей — позже).
2. ~~ARGSORT~~ — bitonic `kernel_argsort_f32_i32`.
3. ~~MUL_MAT_ID~~ — Q4_0 SOA `kernel_mul_mv_id_q4_0_f32_8x_flat` (+ фикс CLAMP workgroup).
4. ~~GATED_DELTA_NET~~ kernel+host — unit 43/43 (`6caa8b9`); **речь 27B отложена** (hybrid состав, не unit).
5. **Координация** — срез с [[nil]] / указатели; OCL-актив не вести в этом такте.

Закрыто по пути 27B до паузы: CPY→view · SSM_CONV · SOFTPLUS (`21f0130`) · GDN (`6caa8b9`).
Отложено: численная/составная поломка hybrid-пути на 27B.
Gemma на ocl: E4B · 12B · **26B-A4B Q4_0 ок** (Hello · 28.1 / 20.9 t/s).

Модели под рукой:
- E4B: arc3 `/tmp/gemma-4-E4B-it-Q4_0.gguf`
- 12B: arc2 `/home/george/models/gguf/gemma-4-12B-it-Q4_0.gguf`
- 26B-A4B: arc3 `/home/george/models/gguf/gemma-4-26B-A4B-it-Q4_0.gguf`
- Qwen3.5-27B: arc3 `/home/george/models/gguf/Qwen3.5-27B-Q6_K.gguf`

RPC: phantom OCL `:18769` · arc3 CUDA `:18767`.
