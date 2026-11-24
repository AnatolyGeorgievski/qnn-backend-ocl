# Milestones · QNN Backend-OCL

| # | goal | note |
|--|--|--|
| 1 | Tensor assembler · QNN ICD on Intel (Arc / xe) | this repo: `src/` + `kernels/` → `ir/` |
| 2 | Same / compatible backend profile for Adreno | separate HW profile; do not mix Arc and Adreno presets |
| — | Quantize BF16/F16 → f8 | tooling outside this repo (`tools/quantize` pointer) |
| — | RPC / fleet | sibling **qnn-backend-rpc** |
| — | GPU mem manager · cache-page align · heap + slices | later: on-demand MoE tensor swap / debug (OPS_LADDER) |
| — | Ops matrix RMS → SSM → QNN/hybrid | OPS_LADDER; IR: OP → cmd_template |

| — | Hybrid: layer on CPU if CPU pulls | BF16/FP16 CPU path only with `AVX*BF16`/`AVX*FP16`; matrix CPU → Nova Lake (next year) |
| — | qnn-backend-cpu (sibling later) | Quants from QuantumNN `qnn.h`/`qnn_gguf.c`; no fake native without ISA |

## Near-term order

1. **Qwen3.5-27B speech** on OCL (B60) — current gate.
2. SCALE bf16 / leftover CPY only if that graph needs them.
3. IR templates · offset cleanup · mem heap/slices — after speech.
4. CPU backend scaffolding when useful; **matrix on CPU** after Nova Lake.

Invariant: Arc path and Adreno path are profiles of one product, not two backends.

## Models under acceptance

Primary: **Qwen** and **Gemma** families (chat marker often a single `<|im_start|>`-class token).

Other architectures: later. TinyLlama-style templates that emit missing special tokens are out of scope for the current gate.
