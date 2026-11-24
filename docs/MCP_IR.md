# MCP + IR · leaving ICD text behind

Goal: stop treating human-readable OpenCL C + vendor ICD JIT as the long-term
execution language. Drive the stack as **graph → Tensor IR → LL-HW**.

| layer | language |
|--|--|
| top | tools / speech / graph construction |
| IR | tensor assembler: `OP · src.dtype(shape) → out` |
| carrier | tensors on a sub-cache · zero-copy where the platform allows |

OpenCL remains a thin transport while Stage-1 speech stays green on Arc.
Stage-2 replaces the middle (OpenCL C → IGC) with IR lowering toward `xe` /
device-specific LL.

Battlemage **B60 / B580** is the training line for Intel; Alchemist A770 stays
in the dual-card matrix. Adreno is a later compatible profile of the same IR,
not an emulator of Adreno on Arc.

See also `ir/` and `ChangeLog` for what already runs on host LL and OpenCL LL.
