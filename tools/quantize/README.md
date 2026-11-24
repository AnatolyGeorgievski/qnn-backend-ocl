# tools/quantize · pointer

Canonical FP8 / type encode tooling does **not** live in this repository.

Look for the QuantumNN project (`qnn.h`, `tools/quantize/`, REPRO notes for FP8).
Do not vend a second copy of `qnn.h` here.

This backend consumes quantized GGUF as llama.cpp does; encode recipes stay upstream of the overlay.
