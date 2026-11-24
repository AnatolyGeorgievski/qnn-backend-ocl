# Milestones · QNN Backend-OCL

| # | goal | note |
|--|--|--|
| 1 | Tensor assembler · QNN ICD on Intel (Arc / xe) | this repo: `src/` + `kernels/` → `ir/` |
| 2 | Same / compatible backend profile for Adreno | separate HW profile; do not mix Arc and Adreno presets |
| — | Quantize BF16/F16 → f8 | tooling outside this repo (`tools/quantize` pointer) |
| — | RPC / fleet | sibling **qnn-backend-rpc** |

Invariant: Arc path and Adreno path are profiles of one product, not two backends.

## Models under acceptance

Primary: **Qwen** and **Gemma** families (chat marker often a single `<|im_start|>`-class token).

Other architectures: later. TinyLlama-style templates that emit missing special tokens are out of scope for the current gate.
