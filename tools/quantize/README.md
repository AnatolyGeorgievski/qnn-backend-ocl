# tools/quantize · BF16/F16 → свой f8

<tags>
- include/qnn.h
- /home/george/qnn-backend-ocl/README.md
- QuantumNN/QUANT.md
- QuantumNN/FPQUANT.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[qnn-backend-ocl]]
path: /home/george/qnn-backend-ocl/tools/quantize/README.md
</topo>

## Назначение

Утилита конвертации весов (BF16/F16) в формат QNN (f8_e4m3 / bf8 + опция remainder/SR). Часть **QNN Backend-OCL**, не зерно знаний QuantumNN.

Выход + отчёт ошибки по каждому тензору. Клиент CLI — следующие коммиты.

**подпись:** Курсор
