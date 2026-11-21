# tools/quantize · указатель

<tags>
- /home/george/QuantumNN/tools/quantize/README.md
- /home/george/QuantumNN/tools/quantize/REPRO-fp8.md
- /home/george/QuantumNN/qnn.h
- /home/george/qnn-backend-ocl/README.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[QuantumNN]]
  - [[qnn-backend-ocl]]
path: /home/george/qnn-backend-ocl/tools/quantize/README.md
</topo>

Канон **не здесь**. Типы, `qnn-encode`, рецепт FP8:

- `/home/george/QuantumNN/qnn.h`
- `/home/george/QuantumNN/tools/quantize/` (`qnn-encode.c`, [REPRO-fp8.md](../../../QuantumNN/tools/quantize/REPRO-fp8.md))
- разбор: `/home/george/QuantumNN/qnn_safetensors.c`

```bash
cd /home/george/QuantumNN/tools/quantize
gcc -O2 -o qnn-encode qnn-encode.c -lm
```

Копию `include/qnn.h` в этом каталоге не вести.

**подпись:** Курсор
